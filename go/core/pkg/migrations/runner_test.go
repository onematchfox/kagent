package migrations

import (
	"context"
	"database/sql"
	"fmt"
	"maps"
	"strings"
	"testing"
	"testing/fstest"
	"time"

	_ "github.com/jackc/pgx/v5/stdlib"
	testcontainers "github.com/testcontainers/testcontainers-go"
	tcpostgres "github.com/testcontainers/testcontainers-go/modules/postgres"
	"github.com/testcontainers/testcontainers-go/wait"
)

// --- migration fixtures ---

// goodCoreFS has two valid core migrations.
var goodCoreFS = fstest.MapFS{
	"core/000001_create.up.sql":   {Data: []byte(`CREATE TABLE mig_test (id SERIAL PRIMARY KEY);`)},
	"core/000001_create.down.sql": {Data: []byte(`DROP TABLE IF EXISTS mig_test;`)},
	"core/000002_alter.up.sql":    {Data: []byte(`ALTER TABLE mig_test ADD COLUMN name TEXT;`)},
	"core/000002_alter.down.sql":  {Data: []byte(`ALTER TABLE mig_test DROP COLUMN IF EXISTS name;`)},
}

// oneCoreFS is just the first migration from goodCoreFS.
var oneCoreFS = fstest.MapFS{
	"core/000001_create.up.sql":   {Data: []byte(`CREATE TABLE mig_test (id SERIAL PRIMARY KEY);`)},
	"core/000001_create.down.sql": {Data: []byte(`DROP TABLE IF EXISTS mig_test;`)},
}

// failOnFirstCoreFS fails immediately on the first migration.
var failOnFirstCoreFS = fstest.MapFS{
	"core/000001_bad.up.sql":   {Data: []byte(`ALTER TABLE no_such_table ADD COLUMN x TEXT;`)},
	"core/000001_bad.down.sql": {Data: []byte(`SELECT 1;`)},
}

// failOnSecondCoreFS succeeds on migration 1 then fails on migration 2.
var failOnSecondCoreFS = fstest.MapFS{
	"core/000001_create.up.sql":   {Data: []byte(`CREATE TABLE mig_test (id SERIAL PRIMARY KEY);`)},
	"core/000001_create.down.sql": {Data: []byte(`DROP TABLE IF EXISTS mig_test;`)},
	"core/000002_bad.up.sql":      {Data: []byte(`ALTER TABLE no_such_table ADD COLUMN x TEXT;`)},
	"core/000002_bad.down.sql":    {Data: []byte(`SELECT 1;`)},
}

// failVectorFS has a vector migration that fails.
var failVectorFS = fstest.MapFS{
	"vector/000001_bad.up.sql":   {Data: []byte(`ALTER TABLE no_such_table ADD COLUMN y TEXT;`)},
	"vector/000001_bad.down.sql": {Data: []byte(`SELECT 1;`)},
}

// expandCoreFS creates shared_data with two columns. Used to test cross-track
// rollback scenarios where the vector track depends on this table.
var expandCoreFS = fstest.MapFS{
	"core/000001_create_shared.up.sql":   {Data: []byte(`CREATE TABLE IF NOT EXISTS shared_data (id SERIAL PRIMARY KEY, col_a TEXT);`)},
	"core/000001_create_shared.down.sql": {Data: []byte(`DROP TABLE IF EXISTS shared_data;`)},
	"core/000002_add_col_b.up.sql":       {Data: []byte(`ALTER TABLE shared_data ADD COLUMN IF NOT EXISTS col_b TEXT;`)},
	"core/000002_add_col_b.down.sql":     {Data: []byte(`ALTER TABLE shared_data DROP COLUMN IF EXISTS col_b;`)},
}

// failVectorWithDependencyFS is a vector migration that partially succeeds
// (adds a column to shared_data) then fails. Its down migration uses IF EXISTS
// so rollback is safe even if the column was never added.
var failVectorWithDependencyFS = fstest.MapFS{
	"vector/000001_bad_depends_on_core.up.sql":   {Data: []byte(`ALTER TABLE shared_data ADD COLUMN IF NOT EXISTS vec_col VECTOR(3); ALTER TABLE no_such_table ADD COLUMN x TEXT;`)},
	"vector/000001_bad_depends_on_core.down.sql": {Data: []byte(`ALTER TABLE shared_data DROP COLUMN IF EXISTS vec_col;`)},
}

// mergeFS combines multiple MapFS values into one.
func mergeFS(fsMaps ...fstest.MapFS) fstest.MapFS {
	out := fstest.MapFS{}
	for _, m := range fsMaps {
		maps.Copy(out, m)
	}
	return out
}

// trackVersion reads the current version from a golang-migrate tracking table.
// Returns 0 if the table is empty or does not exist (fully rolled back).
func trackVersion(t *testing.T, connStr, table string) uint {
	t.Helper()
	db, err := sql.Open("pgx", connStr)
	if err != nil {
		t.Fatalf("trackVersion: open db: %v", err)
	}
	defer db.Close()
	var v uint
	err = db.QueryRowContext(context.Background(),
		fmt.Sprintf(`SELECT version FROM %s LIMIT 1`, table)).Scan(&v)
	if err != nil {
		return 0 // sql.ErrNoRows or table doesn't exist
	}
	return v
}

// startTestDB spins up a pgvector Postgres container and returns its connection
// string, registering cleanup with t. It does not run any migrations.
func startTestDB(t *testing.T) string {
	t.Helper()
	ctx := context.Background()
	pgContainer, err := tcpostgres.Run(ctx,
		"pgvector/pgvector:pg18-trixie",
		tcpostgres.WithDatabase("kagent_test"),
		tcpostgres.WithUsername("postgres"),
		tcpostgres.WithPassword("kagent"),
		testcontainers.WithWaitStrategy(
			wait.ForLog("database system is ready to accept connections").
				WithOccurrence(2).
				WithStartupTimeout(60*time.Second),
		),
	)
	if err != nil {
		t.Fatalf("startTestDB: start container: %v", err)
	}
	t.Cleanup(func() {
		if err := pgContainer.Terminate(ctx); err != nil {
			t.Logf("warning: failed to terminate postgres container: %v", err)
		}
	})
	connStr, err := pgContainer.ConnectionString(ctx, "sslmode=disable")
	if err != nil {
		t.Fatalf("startTestDB: connection string: %v", err)
	}
	return connStr
}

// goodVectorFS has a valid vector migration.
var goodVectorFS = fstest.MapFS{
	"vector/000001_create.up.sql":   {Data: []byte(`CREATE EXTENSION IF NOT EXISTS vector; CREATE TABLE IF NOT EXISTS vec_test (id SERIAL PRIMARY KEY, embedding vector(3));`)},
	"vector/000001_create.down.sql": {Data: []byte(`DROP TABLE IF EXISTS vec_test; DROP EXTENSION IF EXISTS vector;`)},
}

// startTestDBWithoutPgvector spins up a plain Postgres container (no pgvector)
// and returns its connection string, registering cleanup with t.
func startTestDBWithoutPgvector(t *testing.T) string {
	t.Helper()
	ctx := context.Background()
	pgContainer, err := tcpostgres.Run(ctx,
		"postgres:18",
		tcpostgres.WithDatabase("kagent_test"),
		tcpostgres.WithUsername("postgres"),
		tcpostgres.WithPassword("kagent"),
		testcontainers.WithWaitStrategy(
			wait.ForLog("database system is ready to accept connections").
				WithOccurrence(2).
				WithStartupTimeout(60*time.Second),
		),
	)
	if err != nil {
		t.Fatalf("startTestDBWithoutPgvector: start container: %v", err)
	}
	t.Cleanup(func() {
		if err := pgContainer.Terminate(ctx); err != nil {
			t.Logf("warning: failed to terminate postgres container: %v", err)
		}
	})
	connStr, err := pgContainer.ConnectionString(ctx, "sslmode=disable")
	if err != nil {
		t.Fatalf("startTestDBWithoutPgvector: connection string: %v", err)
	}
	return connStr
}

// tableExists checks whether a table exists in the public schema.
func tableExists(t *testing.T, connStr, table string) bool {
	t.Helper()
	db, err := sql.Open("pgx", connStr)
	if err != nil {
		t.Fatalf("tableExists: open db: %v", err)
	}
	defer db.Close()
	var exists bool
	err = db.QueryRowContext(context.Background(),
		"SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = $1)",
		table).Scan(&exists)
	if err != nil {
		t.Fatalf("tableExists: query: %v", err)
	}
	return exists
}

// --- applyDir tests ---

func TestApplyDir_HappyPath(t *testing.T) {
	connStr := startTestDB(t)

	prev, err := applyDir(connStr, goodCoreFS, "core", "schema_migrations")
	if err != nil {
		t.Fatalf("applyDir: %v", err)
	}
	if prev != 0 {
		t.Errorf("prevVersion = %d, want 0", prev)
	}
	if got := trackVersion(t, connStr, "schema_migrations"); got != 2 {
		t.Errorf("version = %d, want 2", got)
	}
}

func TestApplyDir_NoOpWhenAlreadyAtLatest(t *testing.T) {
	connStr := startTestDB(t)

	if _, err := applyDir(connStr, goodCoreFS, "core", "schema_migrations"); err != nil {
		t.Fatalf("first apply: %v", err)
	}
	prev, err := applyDir(connStr, goodCoreFS, "core", "schema_migrations")
	if err != nil {
		t.Fatalf("second apply: %v", err)
	}
	if prev != 2 {
		t.Errorf("prevVersion on no-op = %d, want 2", prev)
	}
	if got := trackVersion(t, connStr, "schema_migrations"); got != 2 {
		t.Errorf("version = %d, want 2", got)
	}
}

func TestApplyDir_NoRollbackWhenFirstMigrationFails(t *testing.T) {
	connStr := startTestDB(t)

	if _, err := applyDir(connStr, failOnFirstCoreFS, "core", "schema_migrations"); err == nil {
		t.Fatal("expected error, got nil")
	}
	// prevVersion was 0 so rollback is skipped to protect pre-existing data.
	// golang-migrate marks version 1 as dirty (the failed migration).
	if got := trackVersion(t, connStr, "schema_migrations"); got != 1 {
		t.Errorf("version after failure = %d, want 1 (dirty, rollback skipped)", got)
	}
}

func TestApplyDir_NoRollbackWhenLaterMigrationFails(t *testing.T) {
	connStr := startTestDB(t)

	if _, err := applyDir(connStr, failOnSecondCoreFS, "core", "schema_migrations"); err == nil {
		t.Fatal("expected error, got nil")
	}
	// Migration 1 succeeded, migration 2 failed. Rollback is skipped because
	// prevVersion was 0. golang-migrate marks version 2 as dirty.
	if got := trackVersion(t, connStr, "schema_migrations"); got != 2 {
		t.Errorf("version after failure = %d, want 2 (dirty, rollback skipped)", got)
	}
}

func TestApplyDir_RollsBackToExistingVersion(t *testing.T) {
	connStr := startTestDB(t)

	// Establish a baseline at version 1.
	if _, err := applyDir(connStr, oneCoreFS, "core", "schema_migrations"); err != nil {
		t.Fatalf("setup: %v", err)
	}

	// Advance to version 2 — should fail and roll back to version 1, not 0.
	if _, err := applyDir(connStr, failOnSecondCoreFS, "core", "schema_migrations"); err == nil {
		t.Fatal("expected error, got nil")
	}
	if got := trackVersion(t, connStr, "schema_migrations"); got != 1 {
		t.Errorf("version after rollback = %d, want 1 (pre-run baseline)", got)
	}
}

// TestApplyDir_RollsBackWithExistingVersion verifies that when migrations have
// previously been applied (prevVersion > 0), rollback always happens on failure.
// This ensures the rollback protection only affects the initial migration run
// (prevVersion == 0), not subsequent upgrades.
func TestApplyDir_RollsBackWithExistingVersion(t *testing.T) {
	connStr := startTestDB(t)

	// Establish a baseline at version 1.
	if _, err := applyDir(connStr, oneCoreFS, "core", "schema_migrations"); err != nil {
		t.Fatalf("setup: %v", err)
	}

	// Verify data exists at version 1.
	if got := trackVersion(t, connStr, "schema_migrations"); got != 1 {
		t.Fatalf("setup: version = %d, want 1", got)
	}

	// Advance to version 2 — should roll back because prevVersion > 0.
	if _, err := applyDir(connStr, failOnSecondCoreFS, "core", "schema_migrations"); err == nil {
		t.Fatal("expected error, got nil")
	}
	if got := trackVersion(t, connStr, "schema_migrations"); got != 1 {
		t.Errorf("version after rollback = %d, want 1 (rollback should happen when prevVersion > 0)", got)
	}
}

// --- cross-version rolldown tests (stored DOWN scripts) ---

// TestApplyDir_RollsDownWhenDBVersionAhead simulates a binary rollback: a newer binary
// (goodCoreFS, v2) applied migrations and stored their DOWN scripts, then an older binary
// (oneCoreFS, v1 max) starts. applyDir should detect the DB is ahead, roll down to v1
// using the stored DOWN script for v2, then run Up (no-op).
func TestApplyDir_RollsDownWhenDBVersionAhead(t *testing.T) {
	connStr := startTestDB(t)

	// Newer binary: apply 2 migrations, storing DOWN scripts for v1 and v2.
	if _, err := applyDir(connStr, goodCoreFS, "core", "schema_migrations"); err != nil {
		t.Fatalf("newer binary apply: %v", err)
	}
	if got := trackVersion(t, connStr, "schema_migrations"); got != 2 {
		t.Fatalf("setup: version = %d, want 2", got)
	}

	// Older binary: only knows about v1. Should roll down from v2 to v1 using stored scripts.
	prev, err := applyDir(connStr, oneCoreFS, "core", "schema_migrations")
	if err != nil {
		t.Fatalf("older binary apply: %v", err)
	}
	if prev != 1 {
		t.Errorf("prevVersion = %d, want 1 (post-rolldown pre-Up version)", prev)
	}
	if got := trackVersion(t, connStr, "schema_migrations"); got != 1 {
		t.Errorf("version after rolldown = %d, want 1", got)
	}
	// The v2 DOWN migration drops the name column; mig_test itself should still exist.
	if !tableExists(t, connStr, "mig_test") {
		t.Error("mig_test table should still exist after rolling down to v1")
	}
}

// TestApplyDir_FailsWhenNoStoredScriptsAndDBVersionAhead verifies that rolldown fails
// with a clear error when stored DOWN scripts are missing (e.g. the newer binary never ran,
// or the table was dropped). This guards against silent data loss.
func TestApplyDir_FailsWhenNoStoredScriptsAndDBVersionAhead(t *testing.T) {
	connStr := startTestDB(t)

	// Newer binary applies 2 migrations and stores DOWN scripts.
	if _, err := applyDir(connStr, goodCoreFS, "core", "schema_migrations"); err != nil {
		t.Fatalf("newer binary apply: %v", err)
	}

	// Simulate the stored DOWN scripts being unavailable.
	db, err := sql.Open("pgx", connStr)
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	defer db.Close()
	if _, err := db.Exec("DELETE FROM migration_down_scripts"); err != nil {
		t.Fatalf("delete down scripts: %v", err)
	}

	// Older binary: no stored DOWN scripts available — should fail with a clear error.
	_, err = applyDir(connStr, oneCoreFS, "core", "schema_migrations")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	if !strings.Contains(err.Error(), "no stored DOWN scripts") {
		t.Errorf("expected 'no stored DOWN scripts' in error, got: %v", err)
	}
	// DB should still be at v2 — rolldown was not attempted.
	if got := trackVersion(t, connStr, "schema_migrations"); got != 2 {
		t.Errorf("version = %d, want 2 (unchanged after failed rolldown)", got)
	}
}

// TestMigrateLockIDMatchesGoMigrate verifies that migrateLockID computes the same
// advisory lock ID as golang-migrate's pgx v5 driver. It does this by holding our
// computed lock on one connection and asserting that a concurrent applyDir (which
// calls golang-migrate's Up() internally) blocks until the lock is released.
//
// If golang-migrate changes its lock ID formula, this test will fail because
// applyDir will complete in the blocked window instead of waiting.
func TestMigrateLockIDMatchesGoMigrate(t *testing.T) {
	connStr := startTestDB(t)

	lockID, err := migrateLockID(connStr, "schema_migrations")
	if err != nil {
		t.Fatalf("compute lock ID: %v", err)
	}

	// Hold the lock on a dedicated connection.
	lockDB, err := sql.Open("pgx", connStr)
	if err != nil {
		t.Fatalf("open lock db: %v", err)
	}
	defer lockDB.Close()
	lockConn, err := lockDB.Conn(context.Background())
	if err != nil {
		t.Fatalf("acquire lock connection: %v", err)
	}
	defer lockConn.Close()

	if _, err := lockConn.ExecContext(context.Background(), "SELECT pg_advisory_lock($1)", lockID); err != nil {
		t.Fatalf("acquire advisory lock: %v", err)
	}

	// Run applyDir in a goroutine. golang-migrate's Up() will try to acquire the same
	// advisory lock and must block until we release ours.
	done := make(chan error, 1)
	go func() {
		_, err := applyDir(connStr, oneCoreFS, "core", "schema_migrations")
		done <- err
	}()

	// Poll pg_locks until we see an ungranted advisory lock, which means the goroutine
	// has reached pg_advisory_lock and is blocked. This is deterministic and adds no
	// unnecessary delay — we release as soon as the wait is confirmed.
	deadline := time.Now().Add(500 * time.Millisecond)
	for {
		if time.Now().After(deadline) {
			t.Fatal("timed out waiting for applyDir to block on advisory lock — lock ID mismatch?")
		}
		select {
		case err := <-done:
			t.Fatalf("applyDir completed without blocking — lock ID mismatch (err: %v)", err)
		default:
		}
		var waiting bool
		_ = lockDB.QueryRow(
			`SELECT EXISTS(SELECT 1 FROM pg_locks WHERE locktype = 'advisory' AND NOT granted)`,
		).Scan(&waiting)
		if waiting {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	// Release our lock — applyDir should now proceed and complete successfully.
	if _, err := lockConn.ExecContext(context.Background(), "SELECT pg_advisory_unlock($1)", lockID); err != nil {
		t.Fatalf("release advisory lock: %v", err)
	}

	select {
	case err := <-done:
		if err != nil {
			t.Errorf("applyDir failed after lock released: %v", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("applyDir did not complete after advisory lock was released")
	}
}

// --- unit tests (no DB required) ---

func TestMaxEmbeddedVersion(t *testing.T) {
	tests := []struct {
		name    string
		fs      fstest.MapFS
		dir     string
		want    uint
		wantErr bool
	}{
		{
			name: "returns highest version across up and down files",
			fs: fstest.MapFS{
				"core/000001_a.up.sql":   {},
				"core/000001_a.down.sql": {},
				"core/000003_b.up.sql":   {},
				"core/000003_b.down.sql": {},
			},
			dir:  "core",
			want: 3,
		},
		{
			name:    "empty dir returns error",
			fs:      fstest.MapFS{"core/.keep": {}},
			dir:     "core",
			wantErr: true,
		},
		{
			name:    "nonexistent dir returns error",
			fs:      fstest.MapFS{},
			dir:     "missing",
			wantErr: true,
		},
		{
			name: "files with non-numeric names are skipped",
			fs: fstest.MapFS{
				"core/README.md":        {},
				"core/000002_x.up.sql":  {},
				"core/000002_x.down.sql": {},
			},
			dir:  "core",
			want: 2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := maxEmbeddedVersion(tt.fs, tt.dir)
			if (err != nil) != tt.wantErr {
				t.Errorf("maxEmbeddedVersion() error = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("maxEmbeddedVersion() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestComputeMigrateLockID(t *testing.T) {
	a := computeMigrateLockID("db", "public", "schema_migrations")
	b := computeMigrateLockID("db", "public", "vector_schema_migrations")
	if a == b {
		t.Error("different tables must produce different lock IDs")
	}
	// Argument-order sanity: swapping dbName and schemaName must change the result.
	c := computeMigrateLockID("public", "db", "schema_migrations")
	if a == c {
		t.Error("swapping dbName and schemaName must change the lock ID")
	}
	// Deterministic: same inputs always produce the same output.
	if got := computeMigrateLockID("db", "public", "schema_migrations"); got != a {
		t.Errorf("computeMigrateLockID not deterministic: got %d, want %d", got, a)
	}
}

// --- rollbackDir tests ---

func TestRollbackDir_RollsBackToTarget(t *testing.T) {
	connStr := startTestDB(t)

	if _, err := applyDir(connStr, goodCoreFS, "core", "schema_migrations"); err != nil {
		t.Fatalf("setup: %v", err)
	}

	rollbackDir(connStr, goodCoreFS, "core", "schema_migrations", 0)

	if got := trackVersion(t, connStr, "schema_migrations"); got != 0 {
		t.Errorf("version after rollback = %d, want 0", got)
	}
}

func TestRollbackDir_PartialRollback(t *testing.T) {
	connStr := startTestDB(t)

	if _, err := applyDir(connStr, goodCoreFS, "core", "schema_migrations"); err != nil {
		t.Fatalf("setup: %v", err)
	}

	// Roll back only one step (2 → 1).
	rollbackDir(connStr, goodCoreFS, "core", "schema_migrations", 1)

	if got := trackVersion(t, connStr, "schema_migrations"); got != 1 {
		t.Errorf("version after partial rollback = %d, want 1", got)
	}
}

// --- cross-track rollback ---

// TestCrossTrackRollback_CoreUnchangedWhenVectorFails covers the case where
// core has no new migrations (ErrNoChange) and vector fails. Core should not
// be downgraded by the cross-track rollback.
func TestCrossTrackRollback_CoreUnchangedWhenVectorFails(t *testing.T) {
	connStr := startTestDB(t)

	combined := mergeFS(goodCoreFS, failVectorFS)

	// Establish core at its latest version before the run.
	if _, err := applyDir(connStr, combined, "core", "schema_migrations"); err != nil {
		t.Fatalf("setup core: %v", err)
	}

	// Core has no new migrations — applyDir returns ErrNoChange.
	corePrev, err := applyDir(connStr, combined, "core", "schema_migrations")
	if err != nil {
		t.Fatalf("core apply (no-op): %v", err)
	}
	if corePrev != 2 {
		t.Fatalf("corePrev = %d, want 2", corePrev)
	}

	// Vector fails and self-rolls-back.
	if _, err := applyDir(connStr, combined, "vector", "vector_schema_migrations"); err == nil {
		t.Fatal("expected vector error, got nil")
	}

	// Cross-track rollback: core should be untouched since corePrev == current version.
	rollbackDir(connStr, combined, "core", "schema_migrations", corePrev)
	if got := trackVersion(t, connStr, "schema_migrations"); got != 2 {
		t.Errorf("core version = %d, want 2 (should not have been downgraded)", got)
	}
}

func TestCrossTrackRollback_CoreRolledBackWhenVectorFails(t *testing.T) {
	connStr := startTestDB(t)

	combined := mergeFS(goodCoreFS, failVectorFS)

	// Core succeeds.
	corePrev, err := applyDir(connStr, combined, "core", "schema_migrations")
	if err != nil {
		t.Fatalf("core apply: %v", err)
	}
	if got := trackVersion(t, connStr, "schema_migrations"); got != 2 {
		t.Fatalf("core version = %d, want 2", got)
	}

	// Vector fails. Self-rollback is skipped because vector prevVersion is 0.
	if _, err := applyDir(connStr, combined, "vector", "vector_schema_migrations"); err == nil {
		t.Fatal("expected vector error, got nil")
	}
	if got := trackVersion(t, connStr, "vector_schema_migrations"); got != 1 {
		t.Errorf("vector version after failure = %d, want 1 (dirty, rollback skipped)", got)
	}

	// Cross-track rollback: core should be rolled back to its pre-run version.
	rollbackDir(connStr, combined, "core", "schema_migrations", corePrev)
	if got := trackVersion(t, connStr, "schema_migrations"); got != corePrev {
		t.Errorf("core version after cross-track rollback = %d, want %d", got, corePrev)
	}
}

// TestCrossTrackRollback_IfExistsGuardsSafeOnVectorFailure verifies that when a
// vector migration fails and triggers a core cross-track rollback, the IF EXISTS
// guards in both down migrations prevent errors even though the vector migration
// only partially applied and shared_data is being dropped by core's rollback.
func TestCrossTrackRollback_IfExistsGuardsSafeOnVectorFailure(t *testing.T) {
	connStr := startTestDB(t)

	combined := mergeFS(expandCoreFS, failVectorWithDependencyFS)

	// Core succeeds (shared_data created with col_a and col_b).
	corePrev, err := applyDir(connStr, combined, "core", "schema_migrations")
	if err != nil {
		t.Fatalf("core apply: %v", err)
	}
	if got := trackVersion(t, connStr, "schema_migrations"); got != 2 {
		t.Fatalf("core version = %d, want 2", got)
	}

	// Vector fails. Self-rollback is skipped because vector prevVersion is 0.
	if _, err := applyDir(connStr, combined, "vector", "vector_schema_migrations"); err == nil {
		t.Fatal("expected vector error, got nil")
	}
	if got := trackVersion(t, connStr, "vector_schema_migrations"); got != 1 {
		t.Errorf("vector version after failure = %d, want 1 (dirty, rollback skipped)", got)
	}

	// Cross-track rollback: core rolls back to its pre-run version.
	rollbackDir(connStr, combined, "core", "schema_migrations", corePrev)
	if got := trackVersion(t, connStr, "schema_migrations"); got != corePrev {
		t.Errorf("core version after cross-track rollback = %d, want %d", got, corePrev)
	}
}

// --- checkPgvector tests ---

func TestCheckPgvector_SucceedsOnPgvectorDB(t *testing.T) {
	connStr := startTestDB(t) // pgvector image
	if err := checkPgvector(connStr); err != nil {
		t.Errorf("checkPgvector on pgvector db: %v", err)
	}
}

func TestCheckPgvector_FailsOnPlainPostgres(t *testing.T) {
	connStr := startTestDBWithoutPgvector(t) // plain postgres image
	if err := checkPgvector(connStr); err == nil {
		t.Error("checkPgvector on plain postgres: expected error, got nil")
	}
}

// --- RunUp end-to-end tests ---

func TestRunUp_CoreAndVector(t *testing.T) {
	connStr := startTestDB(t)
	combined := mergeFS(goodCoreFS, goodVectorFS)

	if err := RunUp(connStr, combined, true); err != nil {
		t.Fatalf("RunUp: %v", err)
	}
	if got := trackVersion(t, connStr, "schema_migrations"); got != 2 {
		t.Errorf("core version = %d, want 2", got)
	}
	if got := trackVersion(t, connStr, "vector_schema_migrations"); got != 1 {
		t.Errorf("vector version = %d, want 1", got)
	}
}

func TestRunUp_CoreOnlyWhenVectorDisabled(t *testing.T) {
	connStr := startTestDB(t)
	combined := mergeFS(goodCoreFS, goodVectorFS)

	if err := RunUp(connStr, combined, false); err != nil {
		t.Fatalf("RunUp: %v", err)
	}
	if got := trackVersion(t, connStr, "schema_migrations"); got != 2 {
		t.Errorf("core version = %d, want 2", got)
	}
	// Vector tracking table should not exist.
	if tableExists(t, connStr, "vector_schema_migrations") {
		t.Error("vector_schema_migrations should not exist when vectorEnabled=false")
	}
}

func TestRunUp_FailsBeforeMigrationsWhenPgvectorMissing(t *testing.T) {
	connStr := startTestDBWithoutPgvector(t)

	err := RunUp(connStr, goodCoreFS, true)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	// Core migrations should NOT have run — no tracking table created.
	if tableExists(t, connStr, "schema_migrations") {
		t.Error("schema_migrations should not exist — pgvector check should fail before any migrations")
	}
}

// TestRunUp_SkipsCoreRollbackWhenVectorFailsOnFirstRun verifies the cross-track
// rollback protection in RunUp: when vector fails and corePrev is 0 (initial run),
// core is not rolled back to protect pre-existing data.
func TestRunUp_SkipsCoreRollbackWhenVectorFailsOnFirstRun(t *testing.T) {
	connStr := startTestDB(t) // pgvector available so checkPgvector passes
	combined := mergeFS(goodCoreFS, failVectorFS)

	err := RunUp(connStr, combined, true)
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	// Core should still be at version 2 — not rolled back to 0.
	if got := trackVersion(t, connStr, "schema_migrations"); got != 2 {
		t.Errorf("core version = %d, want 2 (should not be rolled back when corePrev == 0)", got)
	}
}

// --- dirty state recovery tests ---

// TestApplyDir_DirtyStateRecoveryOnRestart simulates a restart after a failed
// migration left the database in a dirty state. On the second call, prevVersion
// is > 0 (the dirty version), so rollback is enabled. The runner should clear
// the dirty state and roll back to the last clean version.
func TestApplyDir_DirtyStateRecoveryOnRestart(t *testing.T) {
	connStr := startTestDB(t)

	// First run: apply version 1, then version 2 fails. prevVersion is 0, so
	// rollback is skipped. Database left at version 2 dirty.
	if _, err := applyDir(connStr, failOnSecondCoreFS, "core", "schema_migrations"); err == nil {
		t.Fatal("expected error, got nil")
	}
	if got := trackVersion(t, connStr, "schema_migrations"); got != 2 {
		t.Fatalf("after first run: version = %d, want 2 (dirty)", got)
	}

	// Second run (simulating restart): prevVersion is now 2 (dirty). The runner
	// should detect dirty state and attempt to clear it. mg.Up() will fail because
	// the database is dirty, then rollbackToVersion clears dirty to version 1.
	_, err := applyDir(connStr, failOnSecondCoreFS, "core", "schema_migrations")
	if err == nil {
		t.Fatal("expected error on second run, got nil")
	}
	// After rollback clears dirty state, version should be at 1 (last clean).
	if got := trackVersion(t, connStr, "schema_migrations"); got != 1 {
		t.Errorf("after restart: version = %d, want 1 (dirty cleared, rolled back)", got)
	}
}
