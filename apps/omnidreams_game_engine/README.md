# OmniDreams Game Engine

This package owns the reusable map, simulation, input, conditioning,
presentation, and inference runtime used by standalone OmniDreams games.
Games inject an application policy into `InteractiveDriveApp`.

The optional engine configuration is a strict, partial `schema_version: 1`
YAML document covering map loading, model selection, rendering, presentation,
wheel input, and diagnostics. Omitted fields use typed code defaults; explicit
CLI options override YAML values. Relative paths in YAML resolve beside the
configuration file.

Standalone games may author semantic road networks with the engine's
[node-graph map format](NODE_GRAPH_MAP_FORMAT.md). The schema keeps authored
topology, generated surfaces, curb collision geometry, and the derived directed
lane graph as separate runtime concepts.
