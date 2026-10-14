# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from nemoguardrails import LLMRails, RailsConfig
from tests.utils import FakeLLMModel


class RecordingFakeLLMModel(FakeLLMModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompts = []

    async def generate_async(self, prompt, *, stop=None, **kwargs):
        self.prompts.append(prompt)
        return await super().generate_async(prompt, stop=stop, **kwargs)


@pytest.fixture
def config_passthrough():
    return RailsConfig.from_content(
        colang_content="",
        yaml_content="""
        models:
          - type: main
            engine: mock
            model: test-model

        passthrough: true
        """,
    )


@pytest.mark.asyncio
async def test_context_message_not_forwarded_in_passthrough(config_passthrough):
    llm = RecordingFakeLLMModel(responses=["Hello there"])
    rails = LLMRails(config=config_passthrough, llm=llm)

    result = await rails.generate_async(
        messages=[
            {"role": "context", "content": {"user_name": "John"}},
            {"role": "user", "content": "hi"},
        ],
        options={"output_vars": ["user_name"]},
    )

    assert result.response[-1]["content"] == "Hello there"
    assert result.output_data == {"user_name": "John"}
    assert [(m.role, m.content) for m in llm.prompts[0]] == [("user", "hi")]


@pytest.mark.asyncio
async def test_event_message_not_forwarded_in_passthrough(config_passthrough):
    llm = RecordingFakeLLMModel(responses=["Hello there"])
    rails = LLMRails(config=config_passthrough, llm=llm)

    result = await rails.generate_async(
        messages=[
            {"role": "event", "event": {"type": "UserSilent"}},
            {"role": "user", "content": "hi"},
        ]
    )

    assert result == {"role": "assistant", "content": "Hello there"}
    assert [(m.role, m.content) for m in llm.prompts[0]] == [("user", "hi")]


@pytest.mark.asyncio
async def test_history_forwarded_unchanged_in_passthrough(config_passthrough):
    llm = RecordingFakeLLMModel(responses=["Hello there"])
    rails = LLMRails(config=config_passthrough, llm=llm)

    await rails.generate_async(
        messages=[
            {"role": "context", "content": {"user_name": "John"}},
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "how are you?"},
        ]
    )

    assert [(m.role, m.content) for m in llm.prompts[0]] == [
        ("system", "Be brief."),
        ("user", "hi"),
        ("assistant", "Hello!"),
        ("user", "how are you?"),
    ]
