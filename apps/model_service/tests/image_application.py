# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU application exposing an image through the existing T2V CLI hooks."""

from pathlib import Path

import torch
from t2v import T2VApplication, T2VSession

from flashdreams.infra.runner_io import load_first_frame_tensor


class ImageSession(T2VSession):
    """Read an application-owned image path when preparing a fresh session."""

    image_path = None

    def init(self):
        if self.image_path is not None:
            self._image = load_first_frame_tensor(
                self.image_path,
                pixel_height=self._session_desc.video_height,
                pixel_width=self._session_desc.video_width,
                device=torch.device(self._pipeline.device),
                dtype=torch.bfloat16,
            )
        super().init()


class ImageApplication(T2VApplication):
    """Supply an image option without changing the shared application's API."""

    session_type = ImageSession

    def __init__(self, *, defaults, input_image):
        super().__init__(defaults=defaults)
        self.input_image = input_image
        self.image_path = None

    def _configure_argument_parser(self, parser):
        if self.input_image != "none":
            parser.add_argument(
                "--image-path", type=Path, required=self.input_image == "required"
            )

    def _apply_parsed_arguments(self, args):
        self.image_path = getattr(args, "image_path", None)

    def create_session(self, session_desc):
        session = super().create_session(session_desc)
        session.image_path = self.image_path
        return session
