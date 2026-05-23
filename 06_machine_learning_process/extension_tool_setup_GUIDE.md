# Pi Extension Tool Setup Guide

## How to Add Custom Tools That the AI Can Use

This guide shows you how to create **custom tools** for pi that I (the AI agent) can call directly — just like `read`, `write`, `edit`, and `bash`. This is the most powerful way to extend pi.

---

## Quick Reference: Where Things Go

| Location | Scope | Auto-discovered? |
|----------|-------|-----------------|
| `~/.pi/agent/extensions/*.ts` | Global (all projects) | Yes |
| `~/.pi/agent/extensions/*/index.ts` | Global (subdirectory) | Yes |
| `.pi/extensions/*.ts` | Project-local | Yes |
| `.pi/extensions/*/index.ts` | Project-local (subdirectory) | Yes |

After creating/editing an extension, run `/reload` in pi to pick it up.

---

## Level 1: Simple Tool Extension (Single File)

The simplest extension is a single `.ts` file that registers a tool.

### Example: A Notebook Cell Insertion Tool

Create `.pi/extensions/notebook-tools.ts`:

```typescript
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import * as fs from "node:fs";
import * as path from "node:path";

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "insert_notebook_cells",
    label: "Insert Notebook Cells",
    description: "Insert new cells (markdown or code) into a Jupyter notebook at a specific position",
    promptSnippet: "Add cells to Jupyter notebooks by specifying position, type, and content",
    promptGuidelines: [
      "Use insert_notebook_cells when modifying Jupyter .ipynb files instead of using write or edit, which can corrupt JSON structure.",
      "insert_notebook_cells handles JSON parsing and cell insertion automatically, preserving notebook integrity.",
    ],
    parameters: Type.Object({
      notebook_path: Type.String({ description: "Path to the .ipynb file" }),
      position: Type.Integer({ description: "0-indexed position to insert cells at. -1 appends at end." }),
      cells: Type.Array(Type.Object({
        cell_type: Type.Union([Type.Literal("markdown"), Type.Literal("code")]),
        source: Type.String({ description: "Cell content as a string" }),
      }), { description: "Array of cells to insert" }),
    }),
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      const { notebook_path, position, cells } = params;
      const cwd = ctx.cwd;
      const fullPath = path.resolve(cwd, notebook_path.replace(/^@/, ""));

      // Progress update
      onUpdate?.({
        content: [{ type: "text", text: `Reading ${notebook_path}...` }],
      });

      // Read and parse the notebook
      let nb: any;
      try {
        const raw = fs.readFileSync(fullPath, "utf-8");
        nb = JSON.parse(raw);
      } catch (e: any) {
        return {
          content: [{ type: "text", text: `Error reading notebook: ${e.message}` }],
          isError: true,
        };
      }

      // Validate it's a notebook
      if (!nb.cells || !Array.isArray(nb.cells)) {
        return {
          content: [{ type: "text", text: "File does not appear to be a valid Jupyter notebook (no cells array)" }],
          isError: true,
        };
      }

      // Create new cell objects
      const newCells = cells.map((cell) => ({
        cell_type: cell.cell_type,
        metadata: {},
        source: cell.source.split("\n").map((line, i, arr) =>
          i < arr.length - 1 ? line + "\n" : line
        ),
        ...(cell.cell_type === "code"
          ? { execution_count: null, outputs: [] }
          : {}),
      }));

      // Insert at position
      const insertAt = position === -1 ? nb.cells.length : position;
      nb.cells.splice(insertAt, 0, ...newCells);

      // Write back
      fs.writeFileSync(fullPath, JSON.stringify(nb, null, 1));

      return {
        content: [{
          type: "text",
          text: `Successfully inserted ${cells.length} cell(s) at position ${insertAt} in ${notebook_path}.\n` +
                `New total: ${nb.cells.length} cells.`,
        }],
        details: { insertedCount: cells.length, position: insertAt },
      };
    },
  });
}
```

### How it works

1. `pi.registerTool()` makes the tool available to the LLM
2. `parameters` uses **TypeBox** schemas — the LLM sees these descriptions
3. `execute()` is the actual implementation — it runs on your machine
4. `promptSnippet` adds a one-liner to the system prompt so the LLM knows when to use it
5. `promptGuidelines` adds usage hints to the system prompt

### Testing

```bash
# Quick test without auto-discovery
pi -e .pi/extensions/notebook-tools.ts

# Or just restart pi / run /reload if in auto-discovered location
```

Then in conversation: *"Add a markdown cell and a code cell to my notebook at position 5"* — I'll use the tool automatically.

---

## Level 2: Tool That Runs Python Scripts

Many data science tools are better implemented in Python. Here's the pattern:

### Extension: `.pi/extensions/ml-tools.ts`

```typescript
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "bootstrap_mi_stability",
    label: "Bootstrap MI Stability",
    description: "Check mutual information feature stability via bootstrapping. " +
      "Runs a Python script that resamples data and computes MI distribution for each feature.",
    promptSnippet: "Analyze feature stability using bootstrap resampling of mutual information scores",
    promptGuidelines: [
      "Use bootstrap_mi_stability when the user wants to check if MI scores are reliable, " +
      "or when selecting stable features for modeling.",
    ],
    parameters: Type.Object({
      target: Type.String({
        description: "Target column name (e.g., 'target_12m')",
        default: "target_12m",
      }),
      n_bootstrap: Type.Integer({
        description: "Number of bootstrap samples",
        default: 200,
        minimum: 50,
        maximum: 1000,
      }),
      mean_threshold: Type.Number({
        description: "Minimum mean MI for feature selection",
        default: 0.01,
      }),
      cv_threshold: Type.Number({
        description: "Maximum coefficient of variation for stable features",
        default: 0.5,
      }),
    }),
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      const cwd = ctx.cwd;
      const scriptPath = path.join(cwd, "tools", "bootstrap_mi.py");

      // Check script exists
      if (!fs.existsSync(scriptPath)) {
        return {
          content: [{
            type: "text",
            text: "Error: tools/bootstrap_mi.py not found. Create it first.",
          }],
          isError: true,
        };
      }

      onUpdate?.({
        content: [{
          type: "text",
          text: `Running ${params.n_bootstrap} bootstrap samples for target '${params.target}'...`,
        }],
      });

      try {
        const result = await pi.exec("python", [
          scriptPath,
          "--target", params.target,
          "--n_bootstrap", String(params.n_bootstrap),
          "--mean_threshold", String(params.mean_threshold),
          "--cv_threshold", String(params.cv_threshold),
        ], { signal, timeout: 300000 }); // 5 min timeout

        // Parse JSON output from Python
        const data = JSON.parse(result.stdout);

        const output = [
          `Bootstrap MI Stability Analysis Complete`,
          `========================================`,
          `Target: ${params.target}`,
          `Bootstrap samples: ${params.n_bootstrap}`,
          ``,
          `Total features: ${data.total_features}`,
          `Features with mean MI > ${params.mean_threshold}: ${data.above_threshold}`,
          `Stable features (MI > threshold AND CV < ${params.cv_threshold}): ${data.stable_count}`,
          ``,
          `Top 10 Stable Features:`,
          ...data.stable_features.slice(0, 10).map(
            (f: any, i: number) => `  ${i+1}. ${f.name}: MI=${f.mean_mi.toFixed(4)}, CV=${f.cv.toFixed(3)}`
          ),
        ].join("\n");

        return {
          content: [{ type: "text", text: output }],
          details: data,
        };
      } catch (e: any) {
        if (e.killed) {
          return {
            content: [{ type: "text", text: "Bootstrap analysis cancelled." }],
          };
        }
        return {
          content: [{ type: "text", text: `Error: ${e.message}\n${e.stderr || ""}` }],
          isError: true,
        };
      }
    },
  });
}
```

### Companion Python Script: `tools/bootstrap_mi.py`

```python
#!/usr/bin/env python3
"""Bootstrap MI stability analysis - outputs JSON for pi extension."""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

def bootstrap_mi(X, y, discrete_features, n_bootstrap=200, random_state=42):
    np.random.seed(random_state)
    n_samples = len(X)
    mi_bootstrap = np.zeros((n_bootstrap, X.shape[1]))
    for i in range(n_bootstrap):
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        mi_boot = mutual_info_classif(
            X.iloc[idx], y.iloc[idx],
            discrete_features=discrete_features,
            random_state=random_state + i
        )
        mi_bootstrap[i, :] = mi_boot
    return pd.DataFrame(mi_bootstrap, columns=X.columns)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="target_12m")
    parser.add_argument("--n_bootstrap", type=int, default=200)
    parser.add_argument("--mean_threshold", type=float, default=0.01)
    parser.add_argument("--cv_threshold", type=float, default=0.5)
    args = parser.parse_args()

    # Load data
    with pd.HDFStore("../data/assets.h5") as store:
        data = store["engineered_features"]

    target_labels = [f"target_{i}m" for i in [1, 2, 3, 6, 12]]
    targets = data.dropna().loc[:, target_labels]
    features = data.dropna().drop(target_labels, axis=1)
    features.sector = pd.factorize(features.sector)[0]

    cat_cols = ["year", "month", "msize", "age", "sector"]
    discrete_features = [features.columns.get_loc(c) for c in cat_cols]

    y_binary = (targets[args.target] > 0).astype(int)

    # Run bootstrap
    mi_bootstrap = bootstrap_mi(features, y_binary, discrete_features, args.n_bootstrap)

    # Compute summary
    mi_mean = mi_bootstrap.mean()
    mi_std = mi_bootstrap.std()
    mi_cv = mi_std / (mi_mean + 1e-10)

    stable_mask = (mi_mean > args.mean_threshold) & (mi_cv < args.cv_threshold)
    stable_features = [
        {"name": name, "mean_mi": float(mi_mean[name]), "cv": float(mi_cv[name])}
        for name in features.columns[stable_mask]
    ]
    stable_features.sort(key=lambda x: -x["mean_mi"])

    # Output JSON (pi extension reads stdout)
    result = {
        "total_features": len(features.columns),
        "above_threshold": int((mi_mean > args.mean_threshold).sum()),
        "stable_count": len(stable_features),
        "stable_features": stable_features,
    }
    json.dump(result, sys.stdout)

if __name__ == "__main__":
    main()
```

---

## Level 3: Multi-Tool Extension with State

For complex extensions with multiple tools and shared state:

### Directory Structure

```
.pi/extensions/
└── ml-analysis/
    ├── index.ts          # Entry point
    ├── tools/
    │   ├── bootstrap.ts  # Bootstrap MI tool
    │   ├── learning-curve.ts  # Learning curve tool
    │   └── cross-val.ts  # Cross-validation tool
    └── utils.ts          # Shared utilities
```

### `index.ts`

```typescript
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { registerBootstrapTool } from "./tools/bootstrap";
import { registerLearningCurveTool } from "./tools/learning-curve";
import { registerCrossValTool } from "./tools/cross-val";

export default function (pi: ExtensionAPI) {
  // Shared state across tools
  const sharedState = {
    lastResults: null as any,
    dataCache: new Map<string, any>(),
  };

  // Register all tools
  registerBootstrapTool(pi, sharedState);
  registerLearningCurveTool(pi, sharedState);
  registerCrossValTool(pi, sharedState);

  // Register convenience command
  pi.registerCommand("ml-analyze", {
    description: "Run full ML analysis pipeline (bootstrap + learning curves)",
    handler: async (args, ctx) => {
      ctx.ui.notify("Running ML analysis pipeline...", "info");
      // Could trigger tools programmatically here
    },
  });

  // Persist results across reloads
  pi.on("session_start", async (_event, ctx) => {
    for (const entry of ctx.sessionManager.getBranch()) {
      if (entry.type === "custom" && entry.customType === "ml-results") {
        sharedState.lastResults = entry.data;
      }
    }
  });
}
```

---

## Complete Tool Registration Reference

### `pi.registerTool()` Options

```typescript
pi.registerTool({
  // Required
  name: "my_tool",                    // Unique tool name (snake_case)
  label: "My Tool",                   // Display name
  description: "What this tool does", // Full description for the LLM
  parameters: Type.Object({...}),     // TypeBox schema

  // Optional but recommended
  promptSnippet: "Short one-liner for Available tools section",
  promptGuidelines: [
    "Use my_tool when the user asks about X.",
    "my_tool is preferred over bash for X operations.",
  ],

  // Optional execution
  prepareArguments(args) {            // Pre-process args before validation
    return args;
  },

  async execute(toolCallId, params, signal, onUpdate, ctx) {
    // toolCallId: unique ID for this tool call
    // params: validated parameters matching your schema
    // signal: AbortSignal (user pressed Escape)
    // onUpdate: progress callback
    // ctx: ExtensionContext (cwd, ui, sessionManager, etc.)

    // Send progress
    onUpdate?.({
      content: [{ type: "text", text: "Working..." }],
    });

    // Return result
    return {
      content: [{ type: "text", text: "Done!" }],
      details: { /* any data */ },  // Optional metadata
      isError: false,                // Optional, defaults to false
    };
  },

  // Optional custom rendering
  renderCall(args, theme, context) { /* ... */ },
  renderResult(result, options, theme, context) { /* ... */ },
});
```

### TypeBox Schema Reference

```typescript
import { Type } from "typebox";
import { StringEnum } from "@earendil-works/pi-ai";

// Basic types
Type.String({ description: "A string parameter" })
Type.Number({ description: "A number", minimum: 0, maximum: 100 })
Type.Integer({ description: "An integer", default: 10 })
Type.Boolean({ description: "A flag", default: false })

// Enums
StringEnum(["option1", "option2", "option3"])

// Optional fields
Type.Optional(Type.String({ description: "Optional string" }))

// Arrays
Type.Array(Type.String())

// Objects
Type.Object({
  name: Type.String(),
  value: Type.Number(),
})

// Nested
Type.Object({
  cells: Type.Array(Type.Object({
    cell_type: StringEnum(["markdown", "code"]),
    source: Type.String(),
  })),
})
```

---

## Event Handling for Tools

### Intercepting tool calls (permission gates)

```typescript
pi.on("tool_call", async (event, ctx) => {
  // Block dangerous commands
  if (event.toolName === "bash" && event.input.command?.includes("rm -rf")) {
    const ok = await ctx.ui.confirm("Dangerous!", "Allow rm -rf?");
    if (!ok) return { block: true, reason: "Blocked by user" };
  }

  // Modify tool arguments before execution
  if (event.toolName === "my_tool") {
    event.input.path = event.input.path.replace(/^@/, "");
  }
});
```

### Modifying tool results

```typescript
pi.on("tool_result", async (event, ctx) => {
  if (event.toolName === "my_tool") {
    // Add context to the result
    return {
      content: [...event.content, { type: "text", text: "\nNote: Results cached." }],
    };
  }
});
```

---

## Tool Design Best Practices

### 1. Good descriptions = good tool usage

```typescript
// BAD - vague
description: "Analyze data"

// GOOD - specific about what it does and when to use it
description: "Run bootstrap stability analysis on mutual information scores. " +
  "Returns mean MI, standard deviation, and coefficient of variation for each feature. " +
  "Use to identify features with reliable (stable) mutual information scores."
```

### 2. Use promptGuidelines to guide the LLM

```typescript
promptGuidelines: [
  "Use bootstrap_mi_stability when selecting features based on mutual information, " +
    "to verify that high MI scores are not due to random chance.",
  "bootstrap_mi_stability is preferred over manually running Python scripts " +
    "for MI stability analysis, because it handles data loading and JSON output automatically.",
]
```

### 3. Report progress for long-running tools

```typescript
async execute(toolCallId, params, signal, onUpdate, ctx) {
  onUpdate?.({ content: [{ type: "text", text: "Starting analysis..." }] });

  // Long computation
  for (let i = 0; i < 100; i++) {
    if (signal?.aborted) break;
    // ... work ...
    onUpdate?.({ content: [{ type: "text", text: `Progress: ${i+1}%` }] });
  }

  return { content: [{ type: "text", text: "Done!" }] };
}
```

### 4. Handle cancellation gracefully

```typescript
async execute(toolCallId, params, signal, onUpdate, ctx) {
  try {
    const result = await pi.exec("python", ["script.py"], { signal, timeout: 60000 });
    return { content: [{ type: "text", text: result.stdout }] };
  } catch (e: any) {
    if (e.killed || signal?.aborted) {
      return { content: [{ type: "text", text: "Cancelled by user." }] };
    }
    return { content: [{ type: "text", text: `Error: ${e.message}` }], isError: true };
  }
}
```

### 5. Return structured data in `details`

```typescript
return {
  content: [{ type: "text", text: "Human-readable summary" }],
  details: {
    stableFeatures: ["feature1", "feature2"],
    scores: { feature1: 0.05, feature2: 0.03 },
    metadata: { runtime: "2.3s", bootstrapSamples: 200 },
  },
};
```

The `details` field is available to other extensions and can be persisted in the session.

---

## Step-by-Step: Creating Your First Extension

### 1. Create the directory

```bash
mkdir -p .pi/extensions
```

### 2. Create your extension file

```bash
# Choose one:
# .pi/extensions/my-tool.ts        (single file)
# .pi/extensions/my-tool/index.ts  (directory with index)
```

### 3. Write the extension

Use the templates above. Start simple:

```typescript
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

export default function (pi: ExtensionAPI) {
  pi.registerTool({
    name: "hello_world",
    label: "Hello World",
    description: "A test tool that greets someone",
    parameters: Type.Object({
      name: Type.String({ description: "Name to greet" }),
    }),
    async execute(toolCallId, params, signal, onUpdate, ctx) {
      return {
        content: [{ type: "text", text: `Hello, ${params.name}!` }],
      };
    },
  });
}
```

### 4. Test it

```bash
# Method 1: Quick test with -e flag
pi -e .pi/extensions/my-tool.ts

# Method 2: Auto-discovered (restart pi or /reload)
# In pi, type: /reload
```

### 5. Verify it works

In your pi conversation:
- Type: *"Use the hello_world tool to greet Alice"*
- The LLM should call `hello_world` with `name: "Alice"`
- You should see: `"Hello, Alice!"`

### 6. Iterate

Edit the `.ts` file, then `/reload` to pick up changes.

---

## Useful Tool Ideas for This Project

Based on this ML for Trading project, here are tools you could create:

| Tool Name | Description | Python Script |
|-----------|-------------|---------------|
| `insert_notebook_cells` | Add cells to .ipynb files without corrupting JSON | N/A (TypeScript) |
| `bootstrap_mi_stability` | Bootstrap MI stability analysis | `tools/bootstrap_mi.py` |
| `compute_learning_curves` | Plot learning curves for a model | `tools/learning_curves.py` |
| `cross_validate_model` | Run cross-validation with error reporting | `tools/cross_val.py` |
| `plot_efficient_frontier` | Generate efficient frontier visualization | `tools/efficient_frontier.py` |
| `run_backtest` | Run a trading strategy backtest | `tools/backtest.py` |
| `generate_tear_sheet` | Create a pyfolio tear sheet | `tools/tear_sheet.py` |

---

## Troubleshooting

### Extension not loading

1. Check the file is in the right place: `~/.pi/agent/extensions/` or `.pi/extensions/`
2. File must end in `.ts` (or be `index.ts` in a subdirectory)
3. Run `/reload` in pi
4. Start pi with `--verbose` to see loading errors

### Tool not being called by LLM

1. Check `description` is clear and specific
2. Add `promptSnippet` so the tool appears in "Available tools"
3. Add `promptGuidelines` to tell the LLM when to use it
4. Make sure the tool is active: use `/tools` command or check `pi.getActiveTools()`

### Tool throwing errors

1. Check `pi.exec()` command and arguments
2. Verify Python script exists and runs standalone
3. Use `--verbose` flag when starting pi for error details
4. Test the tool directly: `"Call the bootstrap_mi_stability tool with target='target_12m'"`

### TypeScript import errors

- `@earendil-works/pi-coding-agent` types are available automatically
- `typebox` is available automatically
- `@earendil-works/pi-ai` is available for `StringEnum`
- Node.js built-ins (`node:fs`, `node:path`, etc.) are available
- For npm packages, add a `package.json` next to your extension and run `npm install`

---

## Sharing Extensions as Pi Packages

Once your extension works, you can share it:

### As a local package

```bash
pi install file:./my-extension
```

### As an npm package

1. Create `package.json`:
```json
{
  "name": "@yourname/pi-ml-tools",
  "version": "1.0.0",
  "keywords": ["pi-package"],
  "pi": {
    "extensions": ["./extensions"]
  }
}
```

2. Publish: `npm publish`
3. Install: `pi install npm:@yourname/pi-ml-tools`

### As a git package

```bash
pi install git:github.com/yourname/pi-ml-tools
```

---

## Summary

| What | How |
|------|-----|
| **Add a simple tool** | Create `.pi/extensions/my-tool.ts` with `pi.registerTool()` |
| **Add a Python-backed tool** | Create `.ts` extension that calls `pi.exec("python", [...])` |
| **Make the LLM use it** | Add `promptSnippet` and `promptGuidelines` |
| **Handle cancellation** | Check `signal.aborted` and `try/catch` for `pi.exec()` |
| **Persist state** | Use `pi.appendEntry()` and restore in `session_start` |
| **Test** | `pi -e ./my-tool.ts` or `/reload` after placing in extensions dir |
| **Share** | `pi install` from npm, git, or local path |

The key insight: **TypeScript is the glue, Python does the math.** Write your analysis logic in Python, wrap it in a TypeScript extension, and the LLM will use it automatically when appropriate.
