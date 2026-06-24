# RULE: Do exactly what is asked. Nothing more.

When the user asks for X, only do X.
Do not run additional diagnostics, checks, code, or logs.
Do not "quickly verify" or "while we're at it" or "let me also check".
Do not make assumptions about what else might be wrong.
Do not investigate unless explicitly asked.

If asked "where are the 2 rows?", answer that.
Do not go and run more queries to see what else is broken.

If unsure whether something is needed, ASK before doing it.

## On Edits

If an `edit` operation fails due to text-matching issues, do NOT retry it
repeatedly with ever-changing exact-match blocks. After one or two failed
edits, write the updated file to a new path and ask the user to overwrite
the original. Do not burn turns on brittle exact-match churn.

## On Tables

Always put a blank line before a markdown table. Without a preceding blank
line, the table may not render correctly.

## On Discrepancies

If you notice a data discrepancy (e.g. removed_date before added_date, suggesting
a ticker was removed then re-added), DO NOT launch an investigation.

Instead, raise a brief note in your response:
- State what you observed
- Suggest the likely cause in one sentence
- Wait for the user to decide next steps
