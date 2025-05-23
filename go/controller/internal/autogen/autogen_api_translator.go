package autogen

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/kagent-dev/kagent/go/autogen/api"
	autogen_client "github.com/kagent-dev/kagent/go/autogen/client"
	"github.com/kagent-dev/kagent/go/controller/api/v1alpha1"
	common "github.com/kagent-dev/kagent/go/controller/internal/utils"
	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

var (
	// hard-coded array of tools that require a model client
	// this is automatically populated from the parent agent's model client
	toolsProvidersRequiringModelClient = []string{
		"kagent.tools.prometheus.GeneratePromQLTool",
		"kagent.tools.k8s.GenerateResourceTool",
	}
	toolsProvidersRequiringOpenaiApiKey = []string{
		"kagent.tools.docs.QueryTool",
	}
)

type ApiTranslator interface {
	TranslateGroupChatForTeam(
		ctx context.Context,
		team *v1alpha1.Team,
	) (*autogen_client.Team, error)

	TranslateGroupChatForAgent(
		ctx context.Context,
		agent *v1alpha1.Agent,
	) (*autogen_client.Team, error)

	TranslateToolServer(ctx context.Context, toolServer *v1alpha1.ToolServer) (*autogen_client.ToolServer, error)
}

type apiTranslator struct {
	kube               client.Client
	defaultModelConfig types.NamespacedName
}

func (a *apiTranslator) TranslateToolServer(ctx context.Context, toolServer *v1alpha1.ToolServer) (*autogen_client.ToolServer, error) {
	// provider = "kagent.tool_servers.StdioMcpToolServer" || "kagent.tool_servers.SseMcpToolServer"
	provider, toolServerConfig, err := a.translateToolServerConfig(ctx, toolServer.Spec.Config, toolServer.ObjectMeta.Namespace)
	if err != nil {
		return nil, err
	}

	return &autogen_client.ToolServer{
		UserID: common.GetGlobalUserID(),
		Component: api.Component{
			Provider:      provider,
			ComponentType: "tool_server",
			Version:       1,
			Description:   toolServer.Spec.Description,
			Label:         toolServer.Name,
			Config:        api.MustToConfig(toolServerConfig),
		},
	}, nil
}

var aliasMap = map[string]string{
	"npx": "bunx",
	"npm": "bun",
}

func commandAlias(command string) string {
	if alias, ok := aliasMap[command]; ok {
		return alias
	}
	return command
}

func (a *apiTranslator) translateToolServerConfig(ctx context.Context, config v1alpha1.ToolServerConfig, namespace string) (string, *api.ToolServerConfig, error) {
	switch {
	case config.Stdio != nil:
		env, err := makeEnvironmentVariables(ctx, a.kube, config.Stdio.Env, config.Stdio.EnvFrom, namespace)
		if err != nil {
			return "", nil, err
		}

		return "kagent.tool_servers.StdioMcpToolServer", &api.ToolServerConfig{
			StdioMcpServerConfig: &api.StdioMcpServerConfig{
				Command: commandAlias(config.Stdio.Command),
				Args:    config.Stdio.Args,
				Env:     env,
			},
		}, nil
	case config.Sse != nil:
		headers, err := convertMapFromAnytype(config.Sse.Headers)
		if err != nil {
			return "", nil, err
		}
		timeout, err := convertDurationToSeconds(config.Sse.Timeout)
		if err != nil {
			return "", nil, err
		}
		sseReadTimeout, err := convertDurationToSeconds(config.Sse.SseReadTimeout)
		if err != nil {
			return "", nil, err
		}

		return "kagent.tool_servers.SseMcpToolServer", &api.ToolServerConfig{
			SseMcpServerConfig: &api.SseMcpServerConfig{
				URL:            config.Sse.URL,
				Headers:        headers,
				Timeout:        timeout,
				SseReadTimeout: sseReadTimeout,
			},
		}, nil
	}

	return "", nil, fmt.Errorf("unsupported tool server config")
}

// Make the environment variables
func makeEnvironmentVariables(ctx context.Context, kubeClient client.Client, env map[string]string, envFrom []corev1.EnvFromSource, namespace string) (map[string]string, error) {
	if len(envFrom) == 0 {
		return env, nil
	}

	result := make(map[string]string)

	var (
		configMaps = make(map[string]*corev1.ConfigMap)
		secrets    = make(map[string]*corev1.Secret)
	)

	// Process envFrom first then allow env to replace existing values.
	for _, envFrom := range envFrom {
		switch {
		case envFrom.ConfigMapRef != nil:
			cm := envFrom.ConfigMapRef
			name := cm.Name
			configMap, ok := configMaps[name]
			if !ok {
				if kubeClient == nil {
					return result, fmt.Errorf("couldn't get configMap %v/%v, no kubeClient defined", namespace, name)
				}
				optional := cm.Optional != nil && *cm.Optional
				configMap = &corev1.ConfigMap{}
				if err := fetchObjKube(
					ctx,
					kubeClient,
					configMap,
					name,
					namespace,
				); err != nil {
					if errors.IsNotFound(err) && optional {
						// ignore error when marked optional
						continue
					}
					return result, err
				}
				configMaps[name] = configMap
			}

			for k, v := range configMap.Data {
				if len(envFrom.Prefix) > 0 {
					k = envFrom.Prefix + k
				}

				result[k] = v
			}
		case envFrom.SecretRef != nil:
			s := envFrom.SecretRef
			name := s.Name
			secret, ok := secrets[name]
			if !ok {
				if kubeClient == nil {
					return result, fmt.Errorf("couldn't get secret %v/%v, no kubeClient defined", namespace, name)
				}
				optional := s.Optional != nil && *s.Optional
				secret = &corev1.Secret{}
				if err := fetchObjKube(
					ctx,
					kubeClient,
					secret,
					name,
					namespace,
				); err != nil {
					if errors.IsNotFound(err) && optional {
						// ignore error when marked optional
						continue
					}
					return result, err
				}
				secrets[name] = secret
			}

			for k, v := range secret.Data {
				if len(envFrom.Prefix) > 0 {
					k = envFrom.Prefix + k
				}

				result[k] = string(v)
			}
		}
	}

	for k, v := range env {
		result[k] = v
	}

	return result, nil
}

func convertDurationToSeconds(timeout string) (int, error) {
	if timeout == "" {
		return 0, nil
	}
	d, err := time.ParseDuration(timeout)
	if err != nil {
		return 0, err
	}
	return int(d.Seconds()), nil
}

func NewAutogenApiTranslator(
	kube client.Client,
	defaultModelConfig types.NamespacedName,
) ApiTranslator {
	return &apiTranslator{
		kube:               kube,
		defaultModelConfig: defaultModelConfig,
	}
}

func (a *apiTranslator) TranslateGroupChatForAgent(ctx context.Context, agent *v1alpha1.Agent) (*autogen_client.Team, error) {
	stream := true
	if agent.Spec.Stream != nil {
		stream = *agent.Spec.Stream
	}
	opts := defaultTeamOptions()
	opts.stream = stream
	return a.translateGroupChatForAgent(ctx, agent, opts, &tState{})
}

func (a *apiTranslator) TranslateGroupChatForTeam(
	ctx context.Context,
	team *v1alpha1.Team,
) (*autogen_client.Team, error) {
	return a.translateGroupChatForTeam(ctx, team, defaultTeamOptions(), &tState{})
}

type teamOptions struct {
	stream bool
}

const MAX_DEPTH = 10

type tState struct {
	// used to prevent infinite loops
	// The recursion limit is 10
	depth uint8
	// used to enforce DAG
	// The final member of the list will be the "parent" agent
	visitedAgents []string
}

func (s *tState) with(agent *v1alpha1.Agent) *tState {
	s.depth++
	s.visitedAgents = append(s.visitedAgents, agent.Name)
	return s
}

func (t *tState) isVisited(agentName string) bool {
	return slices.Contains(t.visitedAgents, agentName)
}

func defaultTeamOptions() *teamOptions {
	return &teamOptions{
		stream: true,
	}
}

func (a *apiTranslator) translateGroupChatForAgent(
	ctx context.Context,
	agent *v1alpha1.Agent,
	opts *teamOptions,
	state *tState,
) (*autogen_client.Team, error) {

	simpleTeam, err := a.simpleRoundRobinTeam(ctx, agent, agent.Name)
	if err != nil {
		return nil, err
	}
	return a.translateGroupChatForTeam(ctx, simpleTeam, opts, state)
}

func (a *apiTranslator) translateGroupChatForTeam(
	ctx context.Context,
	team *v1alpha1.Team,
	opts *teamOptions,
	state *tState,
) (*autogen_client.Team, error) {
	// get model config
	roundRobinTeamConfig := team.Spec.RoundRobinTeamConfig
	selectorTeamConfig := team.Spec.SelectorTeamConfig
	magenticOneTeamConfig := team.Spec.MagenticOneTeamConfig
	swarmTeamConfig := team.Spec.SwarmTeamConfig

	modelConfigRef := a.defaultModelConfig
	if team.Spec.ModelConfig != "" {
		modelConfigRef = types.NamespacedName{
			Name:      team.Spec.ModelConfig,
			Namespace: team.Namespace,
		}
	}
	modelConfig := &v1alpha1.ModelConfig{}
	err := fetchObjKube(
		ctx,
		a.kube,
		modelConfig,
		modelConfigRef.Name,
		modelConfigRef.Namespace,
	)
	if err != nil {
		return nil, err
	}

	modelClientWithStreaming, err := a.createModelClientForProvider(ctx, modelConfig, true)
	if err != nil {
		return nil, err
	}

	modelClientWithoutStreaming, err := a.createModelClientForProvider(ctx, modelConfig, false)
	if err != nil {
		return nil, err
	}

	modelContext := &api.Component{
		Provider:      "autogen_core.model_context.UnboundedChatCompletionContext",
		ComponentType: "chat_completion_context",
		Version:       1,
		Description:   "An unbounded chat completion context that keeps a view of the all the messages.",
		Label:         "UnboundedChatCompletionContext",
		Config:        map[string]interface{}{},
	}

	var participants []*api.Component

	for _, agentName := range team.Spec.Participants {
		agent := &v1alpha1.Agent{}
		err := fetchObjKube(
			ctx,
			a.kube,
			agent,
			agentName,
			team.Namespace,
		)
		if err != nil {
			return nil, err
		}

		participant, err := a.translateAssistantAgent(
			ctx,
			agent,
			modelConfig,
			modelClientWithStreaming,
			modelClientWithoutStreaming,
			modelContext,
			opts,
			state,
		)
		if err != nil {
			return nil, err
		}

		participants = append(participants, participant)
	}

	if swarmTeamConfig != nil {
		planningAgent := MakeBuiltinPlanningAgent(
			"planning_agent",
			participants,
			modelClientWithStreaming,
		)
		// prepend builtin planning agent when using swarm mode
		participants = append(
			[]*api.Component{planningAgent},
			participants...,
		)
	}

	terminationCondition, err := translateTerminationCondition(team.Spec.TerminationCondition)
	if err != nil {
		return nil, err
	}

	commonTeamConfig := api.CommonTeamConfig{
		Participants: participants,
		Termination:  terminationCondition,
	}

	var teamConfig *api.Component
	if roundRobinTeamConfig != nil {
		teamConfig = &api.Component{
			Provider:      "autogen_agentchat.teams.RoundRobinGroupChat",
			ComponentType: "team",
			Version:       1,
			Description:   team.Spec.Description,
			Config: api.MustToConfig(&api.RoundRobinGroupChatConfig{
				CommonTeamConfig: commonTeamConfig,
			}),
		}
	} else if selectorTeamConfig != nil {
		teamConfig = &api.Component{
			Provider:      "autogen_agentchat.teams.SelectorGroupChat",
			ComponentType: "team",
			Version:       1,
			Description:   team.Spec.Description,
			Config: api.MustToConfig(&api.SelectorGroupChatConfig{
				CommonTeamConfig: commonTeamConfig,
				SelectorPrompt:   selectorTeamConfig.SelectorPrompt,
			}),
		}
	} else if magenticOneTeamConfig != nil {
		teamConfig = &api.Component{
			Provider:      "autogen_agentchat.teams.MagenticOneGroupChat",
			ComponentType: "team",
			Version:       1,
			Description:   team.Spec.Description,
			Config: api.MustToConfig(&api.MagenticOneGroupChatConfig{
				CommonTeamConfig:  commonTeamConfig,
				MaxStalls:         magenticOneTeamConfig.MaxStalls,
				FinalAnswerPrompt: magenticOneTeamConfig.FinalAnswerPrompt,
			}),
		}
	} else if swarmTeamConfig != nil {
		teamConfig = &api.Component{
			Provider:      "autogen_agentchat.teams.SwarmTeam",
			ComponentType: "team",
			Version:       1,
			Description:   team.Spec.Description,
			Config: api.MustToConfig(&api.SwarmTeamConfig{
				CommonTeamConfig: commonTeamConfig,
			}),
		}
	} else {
		return nil, fmt.Errorf("no team config specified")
	}

	teamConfig.Label = team.Name

	return &autogen_client.Team{
		Component: teamConfig,
		BaseObject: autogen_client.BaseObject{
			UserID: common.GetGlobalUserID(), // always use global id
		},
	}, nil
}

func (a *apiTranslator) simpleRoundRobinTeam(ctx context.Context, agent *v1alpha1.Agent, name string) (*v1alpha1.Team, error) {

	modelConfig := a.defaultModelConfig
	// Use the provided model config if set, otherwise use the default one
	if agent.Spec.ModelConfig != "" {
		modelConfig = types.NamespacedName{
			Name:      agent.Spec.ModelConfig,
			Namespace: agent.Namespace,
		}
	}
	if err := a.kube.Get(ctx, modelConfig, &v1alpha1.ModelConfig{}); err != nil {
		return nil, err
	}
	// generate an internal round robin "team" for the society of mind agent
	meta := agent.ObjectMeta.DeepCopy()
	// This is important so we don't output this message in the CLI/UI
	meta.Name = name
	team := &v1alpha1.Team{
		ObjectMeta: *meta,
		TypeMeta: metav1.TypeMeta{
			Kind:       "Team",
			APIVersion: "kagent.dev/v1alpha1",
		},
		Spec: v1alpha1.TeamSpec{
			Participants:         []string{agent.Name},
			Description:          agent.Spec.Description,
			ModelConfig:          agent.Spec.ModelConfig,
			RoundRobinTeamConfig: &v1alpha1.RoundRobinTeamConfig{},
			TerminationCondition: v1alpha1.TerminationCondition{
				TextMessageTermination: &v1alpha1.TextMessageTermination{
					Source: convertToPythonIdentifier(agent.Name),
				},
			},
		},
	}
	return team, nil
}

func (a *apiTranslator) translateAssistantAgent(
	ctx context.Context,
	agent *v1alpha1.Agent,
	modelConfig *v1alpha1.ModelConfig,
	modelClientWithStreaming *api.Component,
	modelClientWithoutStreaming *api.Component,
	modelContext *api.Component,
	opts *teamOptions,
	state *tState,
) (*api.Component, error) {

	tools := []*api.Component{}
	for _, tool := range agent.Spec.Tools {
		switch {
		case tool.Builtin != nil:
			autogenTool, err := a.translateBuiltinTool(
				ctx,
				modelClientWithoutStreaming,
				modelConfig,
				tool.Builtin,
			)
			if err != nil {
				return nil, err
			}
			tools = append(tools, autogenTool)
		case tool.McpServer != nil:
			for _, toolName := range tool.McpServer.ToolNames {
				autogenTool, err := translateToolServerTool(
					ctx,
					a.kube,
					tool.McpServer.ToolServer,
					toolName,
					agent.Namespace,
				)
				if err != nil {
					return nil, err
				}
				tools = append(tools, autogenTool)
			}
		case tool.Agent != nil:
			if tool.Agent.Ref == agent.Name {
				return nil, fmt.Errorf("agent tool cannot be used to reference itself, %s", agent.Name)
			}

			if state.isVisited(tool.Agent.Ref) {
				return nil, fmt.Errorf("cycle detected in agent tool chain: %s -> %s", agent.Name, tool.Agent.Ref)
			}

			if state.depth > MAX_DEPTH {
				return nil, fmt.Errorf("recursion limit reached in agent tool chain: %s -> %s", agent.Name, tool.Agent.Ref)
			}

			// Translate a nested tool
			toolAgent := v1alpha1.Agent{}

			err := fetchObjKube(
				ctx,
				a.kube,
				&toolAgent,
				tool.Agent.Ref,
				agent.Namespace,
			)
			if err != nil {
				return nil, err
			}

			team, err := a.simpleRoundRobinTeam(ctx, &toolAgent, toolAgent.Name)
			if err != nil {
				return nil, err
			}
			autogenTool, err := a.translateGroupChatForTeam(ctx, team, &teamOptions{}, state.with(agent))
			if err != nil {
				return nil, err
			}

			tool := &api.Component{
				Provider:      "autogen_agentchat.tools.TeamTool",
				ComponentType: "tool",
				Version:       1,
				Config: api.MustToConfig(&api.TeamToolConfig{
					Name:        toolAgent.Name,
					Description: toolAgent.Spec.Description,
					Team:        autogenTool.Component,
				}),
			}

			tools = append(tools, tool)

		default:
			return nil, fmt.Errorf("tool must have a provider or tool server")
		}
	}

	sysMsg := agent.Spec.SystemMessage

	cfg := &api.AssistantAgentConfig{
		Name:         convertToPythonIdentifier(agent.Name),
		Tools:        tools,
		ModelContext: modelContext,
		Description:  agent.Spec.Description,
		// TODO(ilackarms): convert to non-ptr with omitempty?
		SystemMessage:         sysMsg,
		ReflectOnToolUse:      false,
		ToolCallSummaryFormat: "\nTool: \n{tool_name}\n\nArguments:\n\n{arguments}\n\nResult: \n{result}\n",
	}

	if opts.stream {
		cfg.ModelClient = modelClientWithStreaming
		cfg.ModelClientStream = true
	} else {
		cfg.ModelClient = modelClientWithoutStreaming
		cfg.ModelClientStream = false
	}

	if agent.Spec.Memory != nil {
		for _, memoryName := range agent.Spec.Memory {
			autogenMemory, err := a.translateMemory(ctx, memoryName, agent.Namespace)
			if err != nil {
				return nil, err
			}

			cfg.Memory = append(cfg.Memory, autogenMemory)
		}
	}

	return &api.Component{
		Provider:      "autogen_agentchat.agents.AssistantAgent",
		ComponentType: "agent",
		Version:       1,
		Description:   agent.Spec.Description,
		Config:        api.MustToConfig(cfg),
	}, nil
}

func (a *apiTranslator) translateMemory(ctx context.Context, memoryName string, memoryNamespace string) (*api.Component, error) {
	memoryObj := &v1alpha1.Memory{}
	err := fetchObjKube(ctx, a.kube, memoryObj, memoryName, memoryNamespace)
	if err != nil {
		return nil, err
	}

	switch memoryObj.Spec.Provider {
	case v1alpha1.Pinecone:
		apiKey, err := a.getMemoryApiKey(ctx, memoryObj)
		if err != nil {
			return nil, err
		}

		threshold, err := strconv.ParseFloat(memoryObj.Spec.Pinecone.ScoreThreshold, 32)
		if err != nil {
			return nil, fmt.Errorf("failed to parse score threshold: %v", err)
		}

		return &api.Component{
			Provider:      "kagent.memory.PineconeMemory",
			ComponentType: "memory",
			Version:       1,
			Config: api.MustToConfig(&api.PineconeMemoryConfig{
				APIKey:         string(apiKey),
				IndexHost:      memoryObj.Spec.Pinecone.IndexHost,
				TopK:           memoryObj.Spec.Pinecone.TopK,
				Namespace:      memoryObj.Spec.Pinecone.Namespace,
				RecordFields:   memoryObj.Spec.Pinecone.RecordFields,
				ScoreThreshold: threshold,
			}),
		}, nil
	}

	return nil, fmt.Errorf("unsupported memory provider: %s", memoryObj.Spec.Provider)
}

func (a *apiTranslator) translateBuiltinTool(
	ctx context.Context,
	modelClient *api.Component,
	modelConfig *v1alpha1.ModelConfig,
	tool *v1alpha1.BuiltinTool,
) (*api.Component, error) {

	toolConfig, err := convertMapFromAnytype(tool.Config)
	if err != nil {
		return nil, err
	}
	// special case where we put the model client in the tool config
	if toolNeedsModelClient(tool.Name) {
		if err := addModelClientToConfig(modelClient, &toolConfig); err != nil {
			return nil, fmt.Errorf("failed to add model client to tool config: %v", err)
		}
	}
	if toolNeedsOpenaiApiKey(tool.Name) {
		if modelConfig.Spec.Provider != v1alpha1.OpenAI {
			return nil, fmt.Errorf("tool %s requires OpenAI API key, but model config is not OpenAI", tool.Name)
		}
		apiKey, err := a.getModelConfigApiKey(ctx, modelConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to get model config api key: %v", err)
		}

		if err := addOpenaiApiKeyToConfig(apiKey, &toolConfig); err != nil {
			return nil, fmt.Errorf("failed to add openai api key to tool config: %v", err)
		}
	}

	providerParts := strings.Split(tool.Name, ".")
	toolLabel := providerParts[len(providerParts)-1]

	return &api.Component{
		Provider:      tool.Name,
		ComponentType: "tool",
		Version:       1,
		Config:        toolConfig,
		Label:         toolLabel,
	}, nil
}

func translateToolServerTool(
	ctx context.Context,
	kube client.Client,
	toolServerName string,
	toolName string,
	agentNamespace string,
) (*api.Component, error) {
	toolServer := &v1alpha1.ToolServer{}
	err := fetchObjKube(
		ctx,
		kube,
		toolServer,
		toolServerName,
		agentNamespace,
	)
	if err != nil {
		return nil, err
	}

	// requires the tool to have been discovered
	for _, discoveredTool := range toolServer.Status.DiscoveredTools {
		if discoveredTool.Name == toolName {
			return convertComponent(discoveredTool.Component)
		}
	}

	return nil, fmt.Errorf("tool %v not found in discovered tools in ToolServer %v", toolName, toolServer.Name)
}

func convertComponent(component v1alpha1.Component) (*api.Component, error) {
	config, err := convertMapFromAnytype(component.Config)
	if err != nil {
		return nil, err
	}
	return &api.Component{
		Provider:         component.Provider,
		ComponentType:    component.ComponentType,
		Version:          component.Version,
		ComponentVersion: component.ComponentVersion,
		Description:      component.Description,
		Label:            component.Label,
		Config:           config,
	}, nil
}

func convertMapFromAnytype(config map[string]v1alpha1.AnyType) (map[string]interface{}, error) {
	// convert to map[string]interface{} to allow kubebuilder schemaless validation
	// see https://github.com/kubernetes-sigs/controller-tools/issues/636 for more info
	// must unmarshal to interface{} to avoid json.RawMessage
	convertedConfig := make(map[string]interface{})

	if config == nil {
		return convertedConfig, nil
	}

	raw, err := json.Marshal(config)
	if err != nil {
		return nil, err
	}

	err = json.Unmarshal(raw, &convertedConfig)
	if err != nil {
		return nil, err
	}

	return convertedConfig, nil
}

func translateTerminationCondition(terminationCondition v1alpha1.TerminationCondition) (*api.Component, error) {
	// ensure only one termination condition is set
	var conditionsSet int
	if terminationCondition.MaxMessageTermination != nil {
		conditionsSet++
	}
	if terminationCondition.TextMentionTermination != nil {
		conditionsSet++
	}
	if terminationCondition.OrTermination != nil {
		conditionsSet++
	}
	if terminationCondition.StopMessageTermination != nil {
		conditionsSet++
	}
	if terminationCondition.TextMessageTermination != nil {
		conditionsSet++
	}
	if conditionsSet != 1 {
		return nil, fmt.Errorf("exactly one termination condition must be set")
	}

	switch {
	case terminationCondition.MaxMessageTermination != nil:
		return &api.Component{
			Provider:      "autogen_agentchat.conditions.MaxMessageTermination",
			ComponentType: "termination",
			Version:       1,
			//ComponentVersion: 1,
			Config: api.MustToConfig(&api.MaxMessageTerminationConfig{
				MaxMessages: terminationCondition.MaxMessageTermination.MaxMessages,
			}),
		}, nil
	case terminationCondition.TextMentionTermination != nil:
		return &api.Component{
			Provider:      "autogen_agentchat.conditions.TextMentionTermination",
			ComponentType: "termination",
			Version:       1,
			//ComponentVersion: 1,
			Config: api.MustToConfig(&api.TextMentionTerminationConfig{
				Text: terminationCondition.TextMentionTermination.Text,
			}),
		}, nil
	case terminationCondition.TextMessageTermination != nil:
		return &api.Component{
			Provider:      "autogen_agentchat.conditions.TextMessageTermination",
			ComponentType: "termination",
			Version:       1,
			//ComponentVersion: 1,
			Config: api.MustToConfig(&api.TextMessageTerminationConfig{
				Source: terminationCondition.TextMessageTermination.Source,
			}),
		}, nil
	case terminationCondition.OrTermination != nil:
		var conditions []*api.Component
		for _, c := range terminationCondition.OrTermination.Conditions {
			subConditon := v1alpha1.TerminationCondition{
				MaxMessageTermination:  c.MaxMessageTermination,
				TextMentionTermination: c.TextMentionTermination,
			}

			condition, err := translateTerminationCondition(subConditon)
			if err != nil {
				return nil, err
			}
			conditions = append(conditions, condition)
		}
		return &api.Component{
			Provider:      "autogen_agentchat.conditions.OrTerminationCondition",
			ComponentType: "termination",
			Version:       1,
			//ComponentVersion: 1,
			Config: api.MustToConfig(&api.OrTerminationConfig{
				Conditions: conditions,
			}),
		}, nil
	case terminationCondition.StopMessageTermination != nil:
		return &api.Component{
			Provider:      "autogen_agentchat.conditions.StopMessageTermination",
			ComponentType: "termination",
			Version:       1,
			//ComponentVersion: 1,
			Config: api.MustToConfig(&api.StopMessageTerminationConfig{}),
			Label:  "StopMessageTermination",
		}, nil
	}

	return nil, fmt.Errorf("unsupported termination condition")
}

func fetchObjKube(ctx context.Context, kube client.Client, obj client.Object, objName, objNamespace string) error {
	ref := getRefFromString(objName, objNamespace)
	err := kube.Get(ctx, ref, obj)
	if err != nil {
		return err
	}
	return nil
}

func convertToPythonIdentifier(name string) string {
	return strings.ReplaceAll(name, "-", "_")
}

func toolNeedsModelClient(provider string) bool {
	for _, p := range toolsProvidersRequiringModelClient {
		if p == provider {
			return true
		}
	}
	return false
}

func toolNeedsOpenaiApiKey(provider string) bool {
	for _, p := range toolsProvidersRequiringOpenaiApiKey {
		if p == provider {
			return true
		}
	}
	return false
}

func addModelClientToConfig(
	modelClient *api.Component,
	toolConfig *map[string]interface{},
) error {
	if *toolConfig == nil {
		*toolConfig = make(map[string]interface{})
	}

	cfg, err := modelClient.ToConfig()
	if err != nil {
		return err
	}

	(*toolConfig)["model_client"] = cfg
	return nil
}

func addOpenaiApiKeyToConfig(
	apiKey []byte,
	toolConfig *map[string]interface{},
) error {
	if *toolConfig == nil {
		*toolConfig = make(map[string]interface{})
	}

	(*toolConfig)["openai_api_key"] = string(apiKey)
	return nil
}

// createModelClientForProvider creates a model client component based on the model provider
func (a *apiTranslator) createModelClientForProvider(ctx context.Context, modelConfig *v1alpha1.ModelConfig, stream bool) (*api.Component, error) {
	switch modelConfig.Spec.Provider {
	case v1alpha1.Anthropic:
		apiKey, err := a.getModelConfigApiKey(ctx, modelConfig)
		if err != nil {
			return nil, err
		}

		config := &api.AnthropicClientConfiguration{
			BaseAnthropicClientConfiguration: api.BaseAnthropicClientConfiguration{
				APIKey:    string(apiKey),
				Model:     modelConfig.Spec.Model,
				ModelInfo: translateModelInfo(modelConfig.Spec.ModelInfo),
			},
		}

		// Add provider-specific configurations
		if modelConfig.Spec.Anthropic != nil {
			anthropicConfig := modelConfig.Spec.Anthropic

			config.BaseURL = anthropicConfig.BaseURL
			if anthropicConfig.MaxTokens > 0 {
				config.MaxTokens = anthropicConfig.MaxTokens
			}

			if anthropicConfig.Temperature != "" {
				temp, err := strconv.ParseFloat(anthropicConfig.Temperature, 64)
				if err == nil {
					config.Temperature = temp
				}
			}

			if anthropicConfig.TopP != "" {
				topP, err := strconv.ParseFloat(anthropicConfig.TopP, 64)
				if err == nil {
					config.TopP = topP
				}
			}

			config.TopK = anthropicConfig.TopK
		}

		// Convert to map
		configMap, err := config.ToConfig()
		if err != nil {
			return nil, fmt.Errorf("failed to convert Anthropic config: %w", err)
		}

		return &api.Component{
			Provider:      "autogen_ext.models.anthropic.AnthropicChatCompletionClient",
			ComponentType: "model",
			Version:       1,
			Config:        configMap,
		}, nil

	case v1alpha1.AzureOpenAI:
		apiKey, err := a.getModelConfigApiKey(ctx, modelConfig)
		if err != nil {
			return nil, err
		}
		config := &api.AzureOpenAIClientConfig{
			BaseOpenAIClientConfig: api.BaseOpenAIClientConfig{
				Model:     modelConfig.Spec.Model,
				APIKey:    string(apiKey),
				ModelInfo: translateModelInfo(modelConfig.Spec.ModelInfo),
			},
		}

		if stream {
			config.StreamOptions = &api.StreamOptions{
				IncludeUsage: true,
			}
		}

		// Add provider-specific configurations
		if modelConfig.Spec.AzureOpenAI != nil {
			azureConfig := modelConfig.Spec.AzureOpenAI

			config.AzureEndpoint = azureConfig.Endpoint
			config.APIVersion = azureConfig.APIVersion
			config.AzureDeployment = azureConfig.DeploymentName
			config.AzureADToken = azureConfig.AzureADToken

			if azureConfig.Temperature != "" {
				temp, err := strconv.ParseFloat(azureConfig.Temperature, 64)
				if err == nil {
					config.Temperature = temp
				}
			}

			if azureConfig.TopP != "" {
				topP, err := strconv.ParseFloat(azureConfig.TopP, 64)
				if err == nil {
					config.TopP = topP
				}
			}
		}

		return &api.Component{
			Provider:      "autogen_ext.models.openai.AzureOpenAIChatCompletionClient",
			ComponentType: "model",
			Version:       1,
			Config:        api.MustToConfig(config),
		}, nil

	case v1alpha1.OpenAI:
		apiKey, err := a.getModelConfigApiKey(ctx, modelConfig)
		if err != nil {
			return nil, err
		}
		config := &api.OpenAIClientConfig{
			BaseOpenAIClientConfig: api.BaseOpenAIClientConfig{
				Model:     modelConfig.Spec.Model,
				APIKey:    string(apiKey),
				ModelInfo: translateModelInfo(modelConfig.Spec.ModelInfo),
			},
		}

		if stream {
			config.StreamOptions = &api.StreamOptions{
				IncludeUsage: true,
			}
		}

		// Add provider-specific configurations
		if modelConfig.Spec.OpenAI != nil {
			openAIConfig := modelConfig.Spec.OpenAI

			if openAIConfig.BaseURL != "" {
				config.BaseURL = &openAIConfig.BaseURL
			}

			if openAIConfig.Organization != "" {
				config.Organization = &openAIConfig.Organization
			}

			if openAIConfig.MaxTokens > 0 {
				config.MaxTokens = openAIConfig.MaxTokens
			}

			if openAIConfig.Temperature != "" {
				temp, err := strconv.ParseFloat(openAIConfig.Temperature, 64)
				if err == nil {
					config.Temperature = temp
				}
			}

			if openAIConfig.TopP != "" {
				topP, err := strconv.ParseFloat(openAIConfig.TopP, 64)
				if err == nil {
					config.TopP = topP
				}
			}

			if openAIConfig.FrequencyPenalty != "" {
				freqP, err := strconv.ParseFloat(openAIConfig.FrequencyPenalty, 64)
				if err == nil {
					config.FrequencyPenalty = freqP
				}
			}

			if openAIConfig.PresencePenalty != "" {
				presP, err := strconv.ParseFloat(openAIConfig.PresencePenalty, 64)
				if err == nil {
					config.PresencePenalty = presP
				}
			}
		}

		return &api.Component{
			Provider:      "autogen_ext.models.openai.OpenAIChatCompletionClient",
			ComponentType: "model",
			Version:       1,
			Config:        api.MustToConfig(config),
		}, nil

	case v1alpha1.Ollama:
		config := &api.OllamaClientConfiguration{
			OllamaCreateArguments: api.OllamaCreateArguments{
				Model: modelConfig.Spec.Model,
				Host:  modelConfig.Spec.Ollama.Host,
			},
			ModelInfo:       translateModelInfo(modelConfig.Spec.ModelInfo),
			FollowRedirects: true,
		}

		if modelConfig.Spec.Ollama != nil {
			ollamaConfig := modelConfig.Spec.Ollama

			if ollamaConfig.Options != nil {
				config.Options = ollamaConfig.Options
			}
		}

		return &api.Component{
			Provider:      "autogen_ext.models.ollama.OllamaChatCompletionClient",
			ComponentType: "model",
			Version:       1,
			Config:        api.MustToConfig(config),
		}, nil

	default:
		return nil, fmt.Errorf("unsupported model provider: %s", modelConfig.Spec.Provider)
	}
}

func translateModelInfo(modelInfo *v1alpha1.ModelInfo) *api.ModelInfo {
	if modelInfo == nil {
		return nil
	}

	return &api.ModelInfo{
		Vision:                 modelInfo.Vision,
		FunctionCalling:        modelInfo.FunctionCalling,
		JSONOutput:             modelInfo.JSONOutput,
		Family:                 modelInfo.Family,
		StructuredOutput:       modelInfo.StructuredOutput,
		MultipleSystemMessages: modelInfo.MultipleSystemMessages,
	}
}

func (a *apiTranslator) getMemoryApiKey(ctx context.Context, memory *v1alpha1.Memory) ([]byte, error) {
	memoryApiKeySecret := &v1.Secret{}
	err := fetchObjKube(
		ctx,
		a.kube,
		memoryApiKeySecret,
		memory.Spec.APIKeySecretRef,
		memory.Namespace,
	)
	if err != nil {
		return nil, err
	}

	if memoryApiKeySecret.Data == nil {
		return nil, fmt.Errorf("memory api key secret data not found")
	}

	memoryApiKey, ok := memoryApiKeySecret.Data[memory.Spec.APIKeySecretKey]
	if !ok {
		return nil, fmt.Errorf("memory api key not found")
	}

	return memoryApiKey, nil
}

func (a *apiTranslator) getModelConfigApiKey(ctx context.Context, modelConfig *v1alpha1.ModelConfig) ([]byte, error) {
	// Only retrieve the secret if APIKeySecretRef is provided
	if modelConfig.Spec.APIKeySecretRef == "" {
		return []byte(""), nil
	}

	modelApiKeySecret := &v1.Secret{}
	err := fetchObjKube(
		ctx,
		a.kube,
		modelApiKeySecret,
		modelConfig.Spec.APIKeySecretRef,
		modelConfig.Namespace,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch API key secret %s/%s: %w", modelConfig.Namespace, modelConfig.Spec.APIKeySecretRef, err)
	}

	if modelApiKeySecret.Data == nil {
		return nil, fmt.Errorf("API key secret %s/%s data not found", modelConfig.Namespace, modelConfig.Spec.APIKeySecretRef)
	}

	modelApiKey, ok := modelApiKeySecret.Data[modelConfig.Spec.APIKeySecretKey]
	if !ok {
		return nil, fmt.Errorf("API key not found in secret %s/%s with key %s", modelConfig.Namespace, modelConfig.Spec.APIKeySecretRef, modelConfig.Spec.APIKeySecretKey)
	}
	return modelApiKey, nil
}

func getRefFromString(ref string, parentNamespace string) types.NamespacedName {
	parts := strings.Split(ref, "/")
	var (
		namespace string
		name      string
	)
	if len(parts) == 2 {
		namespace = parts[0]
		name = parts[1]
	} else {
		namespace = parentNamespace
		name = ref
	}

	return types.NamespacedName{
		Namespace: namespace,
		Name:      name,
	}
}
