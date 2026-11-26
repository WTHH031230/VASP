/**
 * VASP VisComposer harness extension.
 *
 * The host (src/pi_poster/designer.py) drives pi in RPC mode. This extension exposes
 * exactly one submit tool per turn and forces the model to call it, so every turn ends
 * with one structured response, as codex --output-schema guaranteed before:
 *   - submit_poster: first turn (poster_html, style_css, design_state?, critique?, summary)
 *   - submit_revision: visual turns (file_edits, design_state?, critique?, summary)
 * Parameter schemas are the arm's response schemas with $refs inlined, read from
 * $PI_POSTER_SCHEMA_DIR/{submit_poster,submit_revision}.json. The host selects the turn
 * with the command "/poster-turn initial|revision" before each prompt.
 */
import { appendFileSync, readFileSync } from "node:fs";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

const TOOLS = {
	initial: {
		name: "submit_poster",
		description: "Submit the first-turn poster source and design objects to the host.",
	},
	revision: {
		name: "submit_revision",
		description: "Submit this checkpoint's exact source edits and design objects to the host.",
	},
} as const;

export default function (pi: ExtensionAPI) {
	const schemaDir = process.env.PI_POSTER_SCHEMA_DIR;
	if (!schemaDir) throw new Error("PI_POSTER_SCHEMA_DIR is not set");
	let turn: keyof typeof TOOLS = "initial";

	for (const key of Object.keys(TOOLS) as (keyof typeof TOOLS)[]) {
		const spec = TOOLS[key];
		const schema = JSON.parse(readFileSync(join(schemaDir, `${spec.name}.json`), "utf8"));
		pi.registerTool({
			name: spec.name,
			label: spec.name,
			description: spec.description,
			parameters: Type.Unsafe(schema),
			async execute() {
				return {
					content: [{ type: "text", text: "Received by the host." }],
					details: {},
					terminate: true,
				};
			},
		});
	}

	const activate = () => pi.setActiveTools([TOOLS[turn].name]);

	pi.registerCommand("poster-turn", {
		description: "Select the submit tool for the next turn: initial | revision",
		handler: async (args) => {
			const value = (args ?? "").trim();
			if (value !== "initial" && value !== "revision") throw new Error(`unknown turn: ${value}`);
			turn = value;
			activate();
		},
	});

	pi.on("session_start", async () => activate());

	pi.on("before_provider_request", (event) => {
		const payload = event.payload as Record<string, unknown> | undefined;
		const tools = payload?.tools;
		if (!payload || !Array.isArray(tools)) return undefined;
		const name = TOOLS[turn].name;
		const offered = tools.some((tool: any) => tool?.function?.name === name || tool?.name === name);
		if (!offered) throw new Error(`submit tool ${name} is not in the request`);
		// Responses API (openai-responses) names the function at the top level; Chat
		// Completions nests it. Some OpenAI-compatible gateways reject function tools with reasoning on Chat
		// Completions for gpt-5.6-sol, so the provider is configured for Responses.
		const choice = "input" in payload
			? { type: "function", name }
			: { type: "function", function: { name } };
		const tier = process.env.PI_POSTER_SERVICE_TIER;
		const forced = {
			...payload,
			...(tier ? { service_tier: tier } : {}),
			tool_choice: choice,
			parallel_tool_calls: false,
		};
		// Debug only: the request body (no credentials) for diagnosing provider errors.
		const log = process.env.PI_POSTER_PAYLOAD_LOG;
		if (log) appendFileSync(log, JSON.stringify(forced) + "\n");
		return forced;
	});
}
