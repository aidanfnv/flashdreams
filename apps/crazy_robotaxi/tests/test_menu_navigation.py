# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Menu navigation through real Dear ImGui contexts, without desktop input."""

from pathlib import Path

import numpy as np
import pytest

from crazy_robotaxi.controls import (
    GamepadButtonStyle,
    canonical_key,
    load_controls_documents,
)
from crazy_robotaxi.game_selection import GameMapOption, GameRaceCourseOption
from crazy_robotaxi.settings import SettingsDocument
from crazy_robotaxi.ui import CrazyRobotaxiImGuiUILoop, TaxiHudState
from flashdreams.runtime_v2.presentation_manager import PresentationManager
from flashdreams.runtime_v2.user_input_event import (
    FocusUserInputEvent,
    GamepadUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.manual


class _Menus:
    def __init__(
        self, imgui, path: Path, *, size=(320, 200), style: GamepadButtonStyle = "Xbox"
    ):
        self.imgui = imgui
        self.context = imgui.create_context()
        io = imgui.get_io()
        io.display_size = size
        io.set_ini_filename("")
        io.backend_flags |= imgui.BackendFlags_.renderer_has_textures
        self.loop = CrazyRobotaxiImGuiUILoop(width=size[0], height=size[1])
        self.loop.state = TaxiHudState(
            *size,
            None,
            gamepad_button_style=style,
            settings_document=SettingsDocument.load(
                path / "settings.yaml",
                pipeline_config=None,
                width=size[0],
                height=size[1],
            ),
            control_documents=load_controls_documents(path / "controls"),
            map_options=tuple(
                GameMapOption(
                    str(i),
                    f"MAP {i}",
                    path / f"map-{i}.yaml",
                    (GameRaceCourseOption("grand-prix", "start"),),
                )
                for i in range(2)
            ),
        )
        self.loop._presentation_manager = PresentationManager()
        self.ticks = 0

    @property
    def state(self):
        return self.loop.state

    def tick(self, *events):
        batch = UserInputEvents(list(events))
        keys = {
            "up": "up_arrow",
            "down": "down_arrow",
            "left": "left_arrow",
            "right": "right_arrow",
            "enter": "enter",
            "escape": "escape",
        }
        for event in events:
            if isinstance(event, KeyboardUserInputEvent):
                key = canonical_key(event.key)
                if key in keys:
                    self.imgui.get_io().add_key_event(
                        getattr(self.imgui.Key, keys[key]),
                        event.state is KeyboardInputState.PRESSED,
                    )
        self.imgui.new_frame()
        self.loop.step_ui(self.imgui, self.ticks, batch)
        self.imgui.render()
        self.ticks += 1

    def settle(self, count=8):
        for _ in range(count):
            self.tick()

    def key(self, key):
        self.tick(
            KeyboardUserInputEvent(
                timestamp=np.uint64(self.ticks),
                key=key,
                state=KeyboardInputState.PRESSED,
            )
        )
        self.settle(2)
        self.tick(
            KeyboardUserInputEvent(
                timestamp=np.uint64(self.ticks),
                key=key,
                state=KeyboardInputState.RELEASED,
            )
        )
        self.settle(2)

    def controller(self, *pressed, action="state", index=0):
        self.tick(
            GamepadUserInputEvent(
                timestamp=np.uint64(self.ticks),
                action=action,
                index=index,
                controller_id=f"controller-{index}",
                mapping="standard",
                pressed=tuple(i in pressed for i in range(17)),
            )
        )

    def button(self, index):
        self.controller(index)
        self.settle(2)
        self.controller()
        self.settle(2)


@pytest.fixture
def menus(tmp_path):
    imgui = pytest.importorskip("imgui_bundle.imgui")
    contexts = []

    def make(**kwargs):
        menu = _Menus(imgui, tmp_path, **kwargs)
        contexts.append(menu)
        menu.settle()
        return menu

    yield make
    for menu in contexts:
        imgui.destroy_context(menu.context)


@pytest.mark.parametrize("size", [(320, 200), (1280, 720)])
def test_keyboard_focus_confirm_back_and_red_outline(menus, size):
    menu = menus(size=size)
    assert menu.context.nav_id == menu.context.nav_window.get_id("TAXI")
    menu.key("ArrowDown")
    assert menu.context.nav_id == menu.context.nav_window.get_id("RACE")
    assert any(
        vertex.col == 0xFF0000FF
        for draw in menu.imgui.get_draw_data().cmd_lists
        for vertex in draw.vtx_buffer
    )
    menu.key("Enter")
    assert menu.state._selected_game_mode == "race"
    assert menu.state._menu_stage == "map"
    menu.key("Escape")
    assert menu.state._menu_stage == "mode"
    assert not menu.state._exit_requested


@pytest.mark.parametrize(
    "style,confirm,back",
    [("Xbox", 0, 1), ("PlayStation", 0, 1), ("Nintendo Switch", 1, 0)],
)
def test_controller_grid_confirm_and_back(menus, style, confirm, back):
    menu = menus(style=style)
    menu.button(13)
    menu.button(confirm)
    assert menu.state._selected_game_mode == "race"
    assert menu.state._menu_stage == "map"
    menu.button(15)
    menu.button(confirm)
    assert menu.state._selected_map_option.map_id == "1"
    assert menu.state._menu_stage == "course"
    menu.button(back)
    assert menu.state._menu_stage == "map"
    menu.button(back)
    assert menu.state._menu_stage == "mode"


def test_controller_input_is_queued_for_next_frame(menus):
    menu = menus()
    original_focus = menu.context.nav_id
    menu.controller(13)
    assert menu.context.nav_id == original_focus
    assert not menu.imgui.is_key_down(menu.imgui.Key.gamepad_dpad_down)
    menu.tick()
    assert menu.imgui.is_key_down(menu.imgui.Key.gamepad_dpad_down)
    menu.tick()
    assert menu.context.nav_id == menu.context.nav_window.get_id("RACE")


def test_options_directional_navigation_and_popup_cancel(menus):
    menu = menus()
    menu.state._open_options()
    menu.settle(24)
    menu.key("ArrowRight")
    menu.key("Enter")
    assert menu.imgui.is_popup_open("", menu.imgui.PopupFlags_.any_popup_id)
    menu.key("Escape")
    assert menu.state._menu_stage == "options"
    assert not menu.imgui.is_popup_open("", menu.imgui.PopupFlags_.any_popup_id)
    menu.key("Escape")
    assert menu.state._menu_stage == "mode"


def test_controls_capture_consumes_navigation_keys(menus):
    menu = menus()
    menu.state._open_controls()
    menu.state._open_controls_device("keyboard")
    menu.settle(24)
    original_focus = menu.context.nav_id
    menu.state._start_binding_capture("drive_forward", 0)
    menu.tick()
    menu.key("ArrowRight")
    assert menu.state._controls_capture is None
    assert menu.state._controls_draft.drive_forward[0].code == "right"
    assert menu.context.nav_id == original_focus
    menu.state._start_binding_capture("drive_forward", 0)
    menu.tick()
    menu.key("Escape")
    assert menu.state._controls_capture is None
    assert menu.state._controls_device == "keyboard"


def test_controls_navigation_scrolls_and_activates_binding_buttons(menus):
    menu = menus()
    menu.state._open_controls()
    menu.state._open_controls_device("keyboard")
    menu.settle(24)
    for _ in range(6):
        menu.key("ArrowDown")
    assert menu.context.nav_window.scroll.y > 0
    menu.key("ArrowRight")
    menu.key("Enter")
    assert menu.state._controls_capture is not None
    menu.key("Escape")
    assert menu.state._controls_capture is None
    assert menu.state._controls_device == "keyboard"
    menu.key("Escape")
    assert menu.state._controls_device is None
    menu.key("Escape")
    assert menu.state._menu_stage == "mode"


@pytest.mark.parametrize(
    "style,back", [("Xbox", 1), ("PlayStation", 1), ("Nintendo Switch", 0)]
)
def test_controller_back_cancels_binding_capture(menus, style, back):
    menu = menus(style=style)
    menu.controller()
    menu.state._open_controls()
    menu.state._open_controls_device("gamepad")
    menu.settle(24)
    draft = menu.state._controls_draft
    menu.state._start_binding_capture("handbrake", 0)
    menu.button(back)
    assert menu.state._controls_capture is None
    assert menu.state._controls_draft == draft
    assert menu.state._controls_device == "gamepad"
    menu.button(back)
    assert menu.state._controls_device is None


def test_gameplay_does_not_accept_menu_navigation(menus):
    menu = menus()
    menu.state._menu_stage = "game"
    menu.button(13)
    menu.button(0)
    flags = (
        menu.imgui.ConfigFlags_.nav_enable_keyboard
        | menu.imgui.ConfigFlags_.nav_enable_gamepad
    )
    assert not menu.imgui.get_io().config_flags & flags
    assert menu.state._menu_stage == "game"


def test_gamepad_release_disconnect_and_focus_loss(menus):
    menu = menus()
    menu.controller(13)
    menu.controller(13, index=1)
    menu.controller(action="disconnected")
    menu.tick()
    assert menu.imgui.is_key_down(menu.imgui.Key.gamepad_dpad_down)
    menu.controller(index=1)
    menu.tick()
    assert not menu.imgui.is_key_down(menu.imgui.Key.gamepad_dpad_down)
    menu.controller(13)
    menu.tick(FocusUserInputEvent(timestamp=np.uint64(menu.ticks), focused=False))
    menu.tick()
    assert not menu.imgui.is_key_down(menu.imgui.Key.gamepad_dpad_down)
