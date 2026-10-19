# Small-display bitmap font

`5x8.pcf.gz` is the unmodified X.Org `font-misc-misc` 5×8 Unicode bitmap
font, copied from the Debian `xfonts-base` package. Its embedded copyright
property and upstream license both state:

> Public domain font.  Share and enjoy.

Upstream: https://gitlab.freedesktop.org/xorg/font/misc-misc

FreeType loads the eight-pixel bitmap strike directly. ImGui locks that
strike; small menu and HUD text is drawn at its native 5×8 size. There is
no outline rasterization or text scaling for this face.
