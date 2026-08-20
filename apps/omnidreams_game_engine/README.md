# OmniDreams Game Engine

This workspace package owns reusable consumer-game simulation above the
FlashDreams V2 I/O API. It converts timestamped keyboard, wheel/gamepad, and
normalized driver-command events into a simulated trajectory, renders that
trajectory into OmniDreams HD-map conditioning, and attaches frame-synchronized
application state to generated results.

The package deliberately does not import `omnidreams.interactive_drive` or the
legacy `flashdreams.runtime.demo` surface. Interactive Drive remains the
enterprise demo on `main`; Crazy Robotaxi composes this engine inside its V2
game session instead.

The engine is application-neutral. Games implement `GameApplication` to add
rules, dynamic actors, and serializable presentation state. The engine retains
V2 device converters for keyboard, native HID wheels, and browser gamepads.
Concrete V2 client windows and converter selection are not yet supplied by
FlashDreams; that end-to-end gap is tracked in the Crazy Robotaxi feature
ledger.
