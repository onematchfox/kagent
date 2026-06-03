package migrations

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"hash/crc32"
	"io/fs"
	"strings"

	"github.com/golang-migrate/migrate/v4"
	migratepgx "github.com/golang-migrate/migrate/v4/database/pgx/v5"
	"github.com/golang-migrate/migrate/v4/source/iofs"
	_ "github.com/jackc/pgx/v5/stdlib"
	ctrl "sigs.k8s.io/controller-runtime"
)

var log = ctrl.Log.WithName("migrations")

// ensureDownScriptsTableSQL creates the table that stores DOWN migration SQL so that
// older binaries can roll back schema versions applied by newer ones.
//
// This table is intentionally NOT managed by golang-migrate: it must exist before any
// migrations run (bootstrapping paradox — it can't be a migration because it is needed
// to execute migrations). Future schema changes to this table must be made via a new
// ALTER TABLE IF NOT EXISTS constant here, not via a migration file.
const ensureDownScriptsTableSQL = `
CREATE TABLE IF NOT EXISTS migration_down_scripts (
    track   TEXT   NOT NULL,
    version BIGINT NOT NULL,
    sql     TEXT   NOT NULL,
    PRIMARY KEY (track, version)
)`

// RunUp applies all pending migrations for the given FS.
// Before running, it stores DOWN scripts in the database to enable cross-version rollbacks.
// If the database is ahead of this binary's max known version, stored DOWN scripts are used
// to roll back to a compatible state first.
// vectorEnabled controls whether the vector track is also applied.
// Returns an error if any track fails (and attempts rollback of previously applied tracks).
func RunUp(url string, migrationsFS fs.FS, vectorEnabled bool) error {
	if vectorEnabled {
		if err := checkPgvector(url); err != nil {
			return fmt.Errorf("vector migrations require pgvector: %w", err)
		}
	}

	corePrev, err := applyDir(url, migrationsFS, "core", "schema_migrations")
	if err != nil {
		return fmt.Errorf("core migrations: %w", err)
	}

	if vectorEnabled {
		if _, err := applyDir(url, migrationsFS, "vector", "vector_schema_migrations"); err != nil {
			if corePrev == 0 {
				log.Info("vector migration failed; skipping core rollback to version 0 to protect pre-existing data")
			} else {
				log.Info("rolling back core after vector failure", "targetVersion", corePrev)
				rollbackDir(url, migrationsFS, "core", "schema_migrations", corePrev)
			}
			return fmt.Errorf("vector migrations: %w", err)
		}
	}

	return nil
}

// applyDir runs Up for dir.
// It stores DOWN scripts from the embedded FS into the database before running Up, so
// that future (older) binaries can roll back schema versions applied by this one.
// If the database is ahead of this binary's max known version, stored DOWN scripts are
// used to roll it back to a compatible state before Up runs.
// If prevVersion is 0 (no migrations have ever been applied), rollback on failure is
// skipped to avoid dropping pre-existing tables on a GORM-to-golang-migrate upgrade.
// It returns the pre-Up version so the caller can roll back this track if a later
// track fails.
func applyDir(url string, migrationsFS fs.FS, dir, migrationsTable string) (prevVersion uint, err error) {
	// Store DOWN scripts from this binary's embedded FS so that older binaries can
	// roll back schema versions applied here. Non-fatal: a failure only reduces future
	// rollback capability; it does not block the current migration run.
	if storeErr := storeDownScripts(url, dir, migrationsFS); storeErr != nil {
		log.Error(storeErr, "failed to store down scripts; rollback capability may be reduced", "track", dir)
	}

	mg, err := newMigrate(url, migrationsFS, dir, migrationsTable)
	if err != nil {
		return 0, err
	}

	prevVersion, _, err = mg.Version()
	if err != nil && !errors.Is(err, migrate.ErrNilVersion) {
		closeMigrate(dir, mg)
		return 0, fmt.Errorf("get pre-migration version for %s: %w", dir, err)
	}
	// prevVersion == 0 when ErrNilVersion (no migrations applied yet).

	// If the database is ahead of this binary's max known version, roll back using
	// stored DOWN scripts to reach a state this binary understands.
	if maxVer, scanErr := maxEmbeddedVersion(migrationsFS, dir); scanErr == nil && prevVersion > maxVer {
		log.Info("database ahead of binary; rolling down using stored scripts",
			"track", dir, "dbVersion", prevVersion, "binaryMax", maxVer)
		closeMigrate(dir, mg) // release advisory lock before raw SQL rolldown
		if rollErr := rollDownWithStoredScripts(url, dir, migrationsTable, maxVer); rollErr != nil {
			return prevVersion, fmt.Errorf(
				"database schema version %d is ahead of this binary's maximum known version %d "+
					"for track %q and rollback failed: %w",
				prevVersion, maxVer, dir, rollErr,
			)
		}
		prevVersion = maxVer
		mg, err = newMigrate(url, migrationsFS, dir, migrationsTable)
		if err != nil {
			return prevVersion, err
		}
	}
	defer closeMigrate(dir, mg)

	if upErr := mg.Up(); upErr != nil {
		if errors.Is(upErr, migrate.ErrNoChange) {
			return prevVersion, nil
		}
		if prevVersion == 0 {
			log.Info("migration failed; skipping rollback to version 0 to protect pre-existing data", "track", dir)
		} else {
			log.Info("migration failed, attempting rollback", "track", dir, "targetVersion", prevVersion)
			if rbErr := rollbackToVersion(mg, dir, prevVersion); rbErr != nil {
				log.Error(rbErr, "rollback failed", "track", dir)
			} else {
				log.Info("rollback complete", "track", dir, "version", prevVersion)
			}
		}
		return prevVersion, fmt.Errorf("run migrations for %s: %w", dir, upErr)
	}
	return prevVersion, nil
}

// rollbackDir opens a fresh migrate instance and rolls dir back to targetVersion.
// Used to roll back a previously-succeeded track when a later track fails.
func rollbackDir(url string, migrationsFS fs.FS, dir, migrationsTable string, targetVersion uint) {
	mg, err := newMigrate(url, migrationsFS, dir, migrationsTable)
	if err != nil {
		log.Error(err, "rollback failed (open)", "track", dir)
		return
	}
	defer closeMigrate(dir, mg)
	if err := rollbackToVersion(mg, dir, targetVersion); err != nil {
		log.Error(err, "rollback failed", "track", dir)
	} else {
		log.Info("rollback complete", "track", dir, "version", targetVersion)
	}
}

// rollbackToVersion rolls the migration state back to targetVersion.
// It handles the dirty-state cleanup golang-migrate requires after a failed
// Up run before down steps can be applied.
func rollbackToVersion(mg *migrate.Migrate, dir string, targetVersion uint) error {
	currentVersion, dirty, err := mg.Version()
	if err != nil {
		if errors.Is(err, migrate.ErrNilVersion) {
			return nil // nothing was applied; nothing to roll back
		}
		return fmt.Errorf("get version after failure for %s: %w", dir, err)
	}

	if dirty {
		// The failed migration is recorded as dirty at currentVersion.
		// Force to the last clean version so Steps can run.
		cleanVersion := int(currentVersion) - 1
		forceTarget := cleanVersion
		if forceTarget < 1 {
			forceTarget = -1 // negative tells golang-migrate to remove the version record entirely
		}
		if err := mg.Force(forceTarget); err != nil {
			return fmt.Errorf("clear dirty state for %s: %w", dir, err)
		}
		if forceTarget < 0 {
			return nil // first migration failed and was cleared; nothing left to roll back
		}
		currentVersion = uint(cleanVersion)
	}

	steps := int(currentVersion) - int(targetVersion)
	if steps <= 0 {
		return nil
	}
	if err := mg.Steps(-steps); err != nil && !errors.Is(err, migrate.ErrNoChange) {
		return fmt.Errorf("roll back %d step(s) for %s: %w", steps, dir, err)
	}
	return nil
}

// storeDownScripts writes each DOWN migration SQL from migrationsFS into the
// migration_down_scripts table so that older binaries can roll back schema versions
// applied by this one. Safe to call on every startup: existing rows are updated so
// that a bug fix to a DOWN script in a newer patch release takes effect.
func storeDownScripts(url, track string, migrationsFS fs.FS) error {
	db, err := sql.Open("pgx", url)
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}
	defer db.Close()

	ctx := context.Background()
	if _, err := db.ExecContext(ctx, ensureDownScriptsTableSQL); err != nil {
		return fmt.Errorf("ensure down scripts table: %w", err)
	}

	entries, err := fs.ReadDir(migrationsFS, track)
	if err != nil {
		return fmt.Errorf("read migration dir %s: %w", track, err)
	}

	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".down.sql") {
			continue
		}
		var version uint
		if _, scanErr := fmt.Sscanf(e.Name(), "%d", &version); scanErr != nil {
			continue
		}
		content, err := fs.ReadFile(migrationsFS, track+"/"+e.Name())
		if err != nil {
			return fmt.Errorf("read down script %s: %w", e.Name(), err)
		}
		if _, err = db.ExecContext(ctx, `
			INSERT INTO migration_down_scripts (track, version, sql)
			VALUES ($1, $2, $3)
			ON CONFLICT (track, version) DO UPDATE SET sql = EXCLUDED.sql`,
			track, version, string(content),
		); err != nil {
			return fmt.Errorf("store down script %s v%d: %w", track, version, err)
		}
	}
	return nil
}

// migrateLockID returns the advisory lock ID that golang-migrate's pgx v5 driver uses
// for the given migrations table. Acquiring this lock provides mutual exclusion with
// concurrent golang-migrate Up()/Down()/Steps() calls on any pod against the same DB.
//
// The formula replicates the Lock() call in
// github.com/golang-migrate/migrate/v4/database/pgx/v5/pgx.go, which calls:
//
//	database.GenerateAdvisoryLockId(DatabaseName, migrationsSchemaName, migrationsTableName)
//
// joining [schemaName, tableName, dbName] with "\x00" then CRC32-IEEE × salt:
//
//	CRC32(currentSchema + "\x00" + migrationsTable + "\x00" + currentDatabase) * 1486364155
//
// If golang-migrate changes this formula, TestMigrateLockIDMatchesGoMigrate will fail.
func migrateLockID(url, migrationsTable string) (int64, error) {
	db, err := sql.Open("pgx", url)
	if err != nil {
		return 0, fmt.Errorf("open database: %w", err)
	}
	defer db.Close()

	var dbName, schemaName string
	if err := db.QueryRow("SELECT current_database(), current_schema()").Scan(&dbName, &schemaName); err != nil {
		return 0, fmt.Errorf("query database and schema names: %w", err)
	}
	return computeMigrateLockID(dbName, schemaName, migrationsTable), nil
}

// computeMigrateLockID is the pure arithmetic from GenerateAdvisoryLockId, split out
// so the test can verify it independently of a live database connection.
func computeMigrateLockID(dbName, schemaName, migrationsTable string) int64 {
	const advisoryLockIDSalt uint32 = 1486364155
	input := schemaName + "\x00" + migrationsTable + "\x00" + dbName
	sum := crc32.ChecksumIEEE([]byte(input)) * advisoryLockIDSalt
	return int64(sum)
}

// rollDownWithStoredScripts rolls the database back to targetVersion using DOWN scripts
// stored in the migration_down_scripts table. It acquires the same advisory lock as
// golang-migrate so that concurrent Up() calls on other pods block until rolldown
// completes, and rolldown blocks if a migration is already in progress.
func rollDownWithStoredScripts(url, dir, migrationsTable string, targetVersion uint) error {
	lockID, err := migrateLockID(url, migrationsTable)
	if err != nil {
		return fmt.Errorf("compute migration lock ID: %w", err)
	}

	db, err := sql.Open("pgx", url)
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}
	defer db.Close()

	// Single connection so the session-level advisory lock is held for the entire
	// rolldown operation.
	conn, err := db.Conn(context.Background())
	if err != nil {
		return fmt.Errorf("acquire connection: %w", err)
	}
	defer conn.Close()

	ctx := context.Background()

	if _, err := conn.ExecContext(ctx, "SELECT pg_advisory_lock($1)", lockID); err != nil {
		return fmt.Errorf("acquire advisory lock: %w", err)
	}
	defer conn.ExecContext(ctx, "SELECT pg_advisory_unlock($1)", lockID) //nolint:errcheck

	rows, err := conn.QueryContext(ctx,
		`SELECT version, sql FROM migration_down_scripts
		 WHERE track = $1 AND version > $2
		 ORDER BY version DESC`,
		dir, targetVersion)
	if err != nil {
		return fmt.Errorf("query stored down scripts for %s: %w", dir, err)
	}

	type downScript struct {
		version uint
		sql     string
	}
	var scripts []downScript
	for rows.Next() {
		var s downScript
		if err := rows.Scan(&s.version, &s.sql); err != nil {
			rows.Close()
			return fmt.Errorf("scan down script: %w", err)
		}
		scripts = append(scripts, s)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return fmt.Errorf("iterate down scripts: %w", err)
	}

	if len(scripts) == 0 {
		// Under the advisory lock, check whether another pod already completed rolldown.
		// Two old pods can both observe "DB ahead" before either acquires the lock; the
		// second serializes here, finds no scripts (deleted by the first), but the DB is
		// already at targetVersion — so this is success, not an error.
		var currentVer uint
		if err := conn.QueryRowContext(ctx,
			"SELECT version FROM "+migrationsTable+" LIMIT 1").Scan(&currentVer); err == nil && currentVer <= targetVersion {
			log.Info("rolldown already completed by another pod",
				"track", dir, "currentVersion", currentVer, "targetVersion", targetVersion)
			return nil
		}
		return fmt.Errorf("no stored DOWN scripts found for track %q versions > %d — "+
			"the binary that applied those migrations must run successfully at least once "+
			"before this binary can roll them back; "+
			"to recover: redeploy the newer image, wait for it to start cleanly, then redeploy this version",
			dir, targetVersion)
	}

	// migrationsTable is always a package-level constant; not user-controlled.
	for _, s := range scripts {
		log.Info("executing stored down migration", "track", dir, "version", s.version)
		tx, err := conn.BeginTx(ctx, nil)
		if err != nil {
			return fmt.Errorf("begin transaction for down %d/%s: %w", s.version, dir, err)
		}

		if _, err := tx.Exec(s.sql); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("execute down migration %d for %s: %w", s.version, dir, err)
		}

		prevVer := s.version - 1
		if prevVer == 0 {
			if _, err := tx.Exec("DELETE FROM " + migrationsTable); err != nil {
				_ = tx.Rollback()
				return fmt.Errorf("clear version tracking for %s: %w", dir, err)
			}
		} else {
			if _, err := tx.Exec(
				"UPDATE "+migrationsTable+" SET version = $1, dirty = false",
				prevVer,
			); err != nil {
				_ = tx.Rollback()
				return fmt.Errorf("update version tracking for %s: %w", dir, err)
			}
		}

		// Remove the applied script atomically so a restart cannot re-execute it.
		// Down migrations are often not idempotent (e.g. DROP COLUMN without IF EXISTS).
		// storeDownScripts will re-insert it if the newer binary is redeployed.
		if _, err := tx.Exec(
			"DELETE FROM migration_down_scripts WHERE track = $1 AND version = $2",
			dir, s.version,
		); err != nil {
			_ = tx.Rollback()
			return fmt.Errorf("delete applied down script %d/%s: %w", s.version, dir, err)
		}

		if err := tx.Commit(); err != nil {
			return fmt.Errorf("commit down migration %d/%s: %w", s.version, dir, err)
		}
		log.Info("completed stored down migration", "track", dir, "version", s.version)
	}
	return nil
}

// maxEmbeddedVersion scans dir inside migrationsFS and returns the highest
// migration version number found. Version numbers are the leading decimal digits
// in filenames of the form "NNNNNN_description.{up,down}.sql".
// Returns 0 and an error if the directory cannot be read or contains no migrations.
func maxEmbeddedVersion(migrationsFS fs.FS, dir string) (uint, error) {
	entries, err := fs.ReadDir(migrationsFS, dir)
	if err != nil {
		return 0, fmt.Errorf("read migration dir %s: %w", dir, err)
	}
	var max uint
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		var v uint
		if _, scanErr := fmt.Sscanf(e.Name(), "%d", &v); scanErr != nil {
			continue
		}
		if v > max {
			max = v
		}
	}
	if max == 0 {
		return 0, fmt.Errorf("no migration files found in %s", dir)
	}
	return max, nil
}

// checkPgvector verifies that the pgvector extension is available on the database.
// This is called before running vector migrations to fail fast with a clear error
// rather than failing mid-migration and triggering a rollback.
func checkPgvector(url string) error {
	db, err := sql.Open("pgx", url)
	if err != nil {
		return fmt.Errorf("open database: %w", err)
	}
	defer db.Close()

	var available bool
	err = db.QueryRow("SELECT EXISTS(SELECT 1 FROM pg_available_extensions WHERE name = 'vector')").Scan(&available)
	if err != nil {
		return fmt.Errorf("check pgvector availability: %w", err)
	}
	if !available {
		return fmt.Errorf("the pgvector extension is not installed on this PostgreSQL instance; either install pgvector or set --database-vector-enabled=false")
	}
	return nil
}

// newMigrate opens a dedicated database connection and constructs a migrate.Migrate
// for the given dir/table. The caller must call closeMigrate when done.
// Uses sql.Open (pgx stdlib shim) — a single dedicated connection — not a pool,
// because the advisory lock is session-level and must not be shared.
func newMigrate(url string, migrationsFS fs.FS, dir, migrationsTable string) (*migrate.Migrate, error) {
	db, err := sql.Open("pgx", url)
	if err != nil {
		return nil, fmt.Errorf("open database for %s: %w", dir, err)
	}

	src, err := iofs.New(migrationsFS, dir)
	if err != nil {
		return nil, fmt.Errorf("load migration files from %s: %w", dir, err)
	}

	driver, err := migratepgx.WithInstance(db, &migratepgx.Config{
		MigrationsTable: migrationsTable,
	})
	if err != nil {
		return nil, fmt.Errorf("create migration driver for %s: %w", dir, err)
	}

	mg, err := migrate.NewWithInstance("iofs", src, "postgres", driver)
	if err != nil {
		return nil, fmt.Errorf("create migrator for %s: %w", dir, err)
	}
	return mg, nil
}

// closeMigrate closes mg, logging source and database close errors separately.
func closeMigrate(dir string, mg *migrate.Migrate) {
	srcErr, dbErr := mg.Close()
	if srcErr != nil {
		log.Error(srcErr, "closing migration source", "track", dir)
	}
	if dbErr != nil {
		log.Error(dbErr, "closing migration database", "track", dir)
	}
}
