# External source snapshots

The v108 generator depends on an upstream source checkout that is not present
in this workspace. Set `HW_V108_SOURCE` to that checkout and, when its core
package is not named `project_core`, set `HW_V108_CORE_PACKAGE` as well. The
launcher fails immediately with a precise missing-source or missing-module
message instead of relying on a machine-specific path.
