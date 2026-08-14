# OmniDreams Game Engine

This workspace package owns reusable consumer-game simulation above the
FlashDreams runtime API. It converts canonical player controls into a simulated
trajectory, renders that trajectory into OmniDreams HD-map conditioning, and
attaches frame-synchronized application state to generated results.

The package deliberately does not import `omnidreams.interactive_drive` or the
legacy `flashdreams.runtime.demo` surface. Interactive Drive remains the
enterprise demo on `main`; games compose this engine inside a
`IFlashDreamsApplicationSession` instead.

The engine is application-neutral. Games implement `GameApplication` to add
rules, dynamic actors, and serializable presentation state. The engine retains
device converters for keyboard, native HID wheels, and browser gamepads, while
the current application milestone consumes the new host's stock
`driver_command` modality. End-to-end wheel integration is explicitly blocked
and tracked in the Crazy Robotaxi feature ledger.
