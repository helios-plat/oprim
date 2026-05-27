# SELF_CHECK P7-B2 — oprim Video Prompt Primitives + Frame Transition + Story Predict

**Date**: 2026-05-27  
**Branch**: main  
**Repo**: /home/soffy/projects/platform/oprim

---

## 1. Checklist

| 项 | 状态 |
|---|---|
| `oprim.style_marker_prompt` — 7 styles, pure fn | ✅ |
| `oprim.lighting_control_prompt` — 6 lightings, pure fn | ✅ |
| `oprim.camera_motion_prompt` — 8 motions + intensity [0,1], pure fn | ✅ |
| `oprim.first_last_frame_transition` — ProviderRegistry injection, FrameTransitionError | ✅ |
| `oprim.video_edit_element_remove` — ProviderRegistry injection, VideoEditError | ✅ |
| `oprim.story_predict` — LLMCaller Protocol + Pydantic StoryPrediction | ✅ |
| `oprim._providers.longcat_avatar` — invoke_local subprocess + invoke_cloud TECHNICAL_DEBT | ✅ |
| `oprim/__init__.py` — 6 public modules + types exported | ✅ |
| `CHANGELOG.md` — P7-B2 entry added | ✅ |
| 测试 ≥43（硬要求） | ✅ 70 |
| 覆盖率 ≥95% | ✅ 96.36% |
| mypy --strict 0 errors | ✅ |
| ruff 0 errors | ✅ |
| 边界严格（只改 oprim，不碰 hevi / obase） | ✅ |

---

## 2. 每元素测试数 vs 要求

| 元素 | 要求 | 实际 |
|---|---|---|
| `style_marker_prompt` | ≥7 | **13** |
| `lighting_control_prompt` | ≥6 | **11** |
| `camera_motion_prompt` | ≥8 | **21** |
| `first_last_frame_transition` | ≥5 | **6** |
| `video_edit_element_remove` | ≥5 | **6** |
| `story_predict` | ≥6 | **7** |
| `_providers.longcat_avatar` | ≥6 | **6** |
| **总计** | **≥43** | **70** |

---

## 3. 测试运行输出

```
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.0.3, pluggy-1.6.0
asyncio: mode=Mode.AUTO

collected 70 items

tests/test_style_marker_prompt.py::test_all_seven_styles_contain_correct_keywords[科普-科普风格-通俗易懂] PASSED
tests/test_style_marker_prompt.py::test_all_seven_styles_contain_correct_keywords[严肃-严肃风格-庄重权威] PASSED
tests/test_style_marker_prompt.py::test_all_seven_styles_contain_correct_keywords[搞笑-搞笑风格-轻松幽默] PASSED
tests/test_style_marker_prompt.py::test_all_seven_styles_contain_correct_keywords[治愈-治愈风格-温暖柔和] PASSED
tests/test_style_marker_prompt.py::test_all_seven_styles_contain_correct_keywords[悬疑-悬疑风格-紧张神秘] PASSED
tests/test_style_marker_prompt.py::test_all_seven_styles_contain_correct_keywords[热血-热血风格-激昂澎湃] PASSED
tests/test_style_marker_prompt.py::test_all_seven_styles_contain_correct_keywords[温暖-温暖风格-亲切温馨] PASSED
tests/test_style_marker_prompt.py::test_base_prompt_appears_first PASSED
tests/test_style_marker_prompt.py::test_unknown_style_raises_value_error PASSED
tests/test_style_marker_prompt.py::test_empty_base_prompt_raises_value_error PASSED
tests/test_style_marker_prompt.py::test_long_base_prompt_not_truncated PASSED
tests/test_style_marker_prompt.py::test_unicode_base_prompt_handled_correctly PASSED
tests/test_style_marker_prompt.py::test_output_format_is_comma_separated_fixed_order PASSED
tests/test_lighting_control_prompt.py::test_all_six_lightings_contain_correct_descriptor[暖-warm and cozy light] PASSED
tests/test_lighting_control_prompt.py::test_all_six_lightings_contain_correct_descriptor[冷-cool and crisp light] PASSED
tests/test_lighting_control_prompt.py::test_all_six_lightings_contain_correct_descriptor[戏剧-dramatic chiaroscuro lighting] PASSED
tests/test_lighting_control_prompt.py::test_all_six_lightings_contain_correct_descriptor[自然-soft natural daylight] PASSED
tests/test_lighting_control_prompt.py::test_all_six_lightings_contain_correct_descriptor[高对比-high contrast hard light] PASSED
tests/test_lighting_control_prompt.py::test_all_six_lightings_contain_correct_descriptor[柔和-soft diffused light] PASSED
tests/test_lighting_control_prompt.py::test_base_prompt_embedded_first PASSED
tests/test_lighting_control_prompt.py::test_unknown_lighting_raises_value_error PASSED
tests/test_lighting_control_prompt.py::test_empty_base_prompt_raises_value_error PASSED
tests/test_lighting_control_prompt.py::test_output_contains_lighting_prefix PASSED
tests/test_lighting_control_prompt.py::test_single_lighting_value_no_combination PASSED
tests/test_camera_motion_prompt.py::test_all_eight_motion_types[pan_left-camera pans left] PASSED
tests/test_camera_motion_prompt.py::test_all_eight_motion_types[pan_right-camera pans right] PASSED
tests/test_camera_motion_prompt.py::test_all_eight_motion_types[tilt_up-camera tilts up] PASSED
tests/test_camera_motion_prompt.py::test_all_eight_motion_types[tilt_down-camera tilts down] PASSED
tests/test_camera_motion_prompt.py::test_all_eight_motion_types[dolly_in-camera moves forward (dolly in)] PASSED
tests/test_camera_motion_prompt.py::test_all_eight_motion_types[dolly_out-camera pulls back (dolly out)] PASSED
tests/test_camera_motion_prompt.py::test_all_eight_motion_types[rotate-camera rotates around subject] PASSED
tests/test_camera_motion_prompt.py::test_all_eight_motion_types[static-static locked-off shot] PASSED
tests/test_camera_motion_prompt.py::test_intensity_mapping[0.0-slow] PASSED
tests/test_camera_motion_prompt.py::test_intensity_mapping[0.33-slow] PASSED
tests/test_camera_motion_prompt.py::test_intensity_mapping[0.34-medium] PASSED
tests/test_camera_motion_prompt.py::test_intensity_mapping[0.5-medium] PASSED
tests/test_camera_motion_prompt.py::test_intensity_mapping[0.67-medium] PASSED
tests/test_camera_motion_prompt.py::test_intensity_mapping[0.68-fast] PASSED
tests/test_camera_motion_prompt.py::test_intensity_mapping[1.0-fast] PASSED
tests/test_camera_motion_prompt.py::test_base_motion_none_format PASSED
tests/test_camera_motion_prompt.py::test_base_motion_non_empty_join_format PASSED
tests/test_camera_motion_prompt.py::test_unknown_motion_type_raises PASSED
tests/test_camera_motion_prompt.py::test_intensity_below_zero_raises PASSED
tests/test_camera_motion_prompt.py::test_intensity_above_one_raises PASSED
tests/test_camera_motion_prompt.py::test_output_not_empty PASSED
tests/test_first_last_frame_transition.py::TestFirstLastFrameTransition::test_success PASSED
tests/test_first_last_frame_transition.py::TestFirstLastFrameTransition::test_provider_not_found PASSED
tests/test_first_last_frame_transition.py::TestFirstLastFrameTransition::test_first_frame_not_found PASSED
tests/test_first_last_frame_transition.py::TestFirstLastFrameTransition::test_last_frame_not_found PASSED
tests/test_first_last_frame_transition.py::TestFirstLastFrameTransition::test_provider_failure PASSED
tests/test_first_last_frame_transition.py::TestFirstLastFrameTransition::test_no_output_produced PASSED
tests/test_video_edit_element_remove.py::TestVideoEditElementRemove::test_mask_as_path_success PASSED
tests/test_video_edit_element_remove.py::TestVideoEditElementRemove::test_mask_as_string_description PASSED
tests/test_video_edit_element_remove.py::TestVideoEditElementRemove::test_video_not_found PASSED
tests/test_video_edit_element_remove.py::TestVideoEditElementRemove::test_provider_not_found PASSED
tests/test_video_edit_element_remove.py::TestVideoEditElementRemove::test_no_output_produced PASSED
tests/test_video_edit_element_remove.py::TestVideoEditElementRemove::test_provider_failure_wrapped PASSED
tests/test_story_predict.py::TestStoryPredict::test_forward_direction PASSED
tests/test_story_predict.py::TestStoryPredict::test_backward_direction PASSED
tests/test_story_predict.py::TestStoryPredict::test_both_directions PASSED
tests/test_story_predict.py::TestStoryPredict::test_custom_prediction_points PASSED
tests/test_story_predict.py::TestStoryPredict::test_reference_image_not_found PASSED
tests/test_story_predict.py::TestStoryPredict::test_llm_returns_non_json_raises_story_predict_error PASSED
tests/test_story_predict.py::TestStoryPredict::test_pydantic_validation_failure_raises_story_predict_error PASSED
tests/_providers/test_longcat_avatar.py::TestInvokeLocal::test_vendor_dir_not_found_raises_setup_error PASSED
tests/_providers/test_longcat_avatar.py::TestInvokeLocal::test_script_not_found_raises_setup_error PASSED
tests/_providers/test_longcat_avatar.py::TestInvokeLocal::test_success_mock_subprocess PASSED
tests/_providers/test_longcat_avatar.py::TestInvokeLocal::test_subprocess_nonzero_exit_raises_error PASSED
tests/_providers/test_longcat_avatar.py::TestInvokeLocal::test_subprocess_timeout_raises_error PASSED
tests/_providers/test_longcat_avatar.py::TestInvokeCloud::test_invoke_cloud_raises_not_implemented PASSED

70 passed in 1.09s
```

---

## 4. 覆盖率

```
Name                                            Stmts   Miss Branch BrPart  Cover   Missing
-------------------------------------------------------------------------------------------
oprim/_providers/longcat_avatar.py                 36      4     10      2    87%   83, 85, 122-123
oprim/camera_motion_prompt/__init__.py             22      0     10      0   100%
oprim/first_last_frame_transition/__init__.py      25      1      6      0    97%   92
oprim/lighting_control_prompt/__init__.py          13      0      4      0   100%
oprim/story_predict/__init__.py                    43      0      6      0   100%
oprim/style_marker_prompt/__init__.py              13      0      4      0   100%
oprim/video_edit_element_remove/__init__.py        24      1      4      0    96%   88
-------------------------------------------------------------------------------------------
TOTAL                                             176      6     44      2    96%
Required test coverage of 95.0% reached. Total coverage: 96.36%
```

**longcat_avatar 未覆盖行说明**:
- 83, 85: `proc.stdout.read()` / `proc.stderr.read()` after `communicate()` — 正常路径下 communicate() 返回 tuple，不经过这两行
- 122-123: `invoke_cloud` 函数体 — `NotImplementedError` TECHNICAL_DEBT stub，test 验证会抛出但内部两行不被 coverage 计为执行

---

## 5. mypy --strict

```
Success: no issues found in 7 source files
```

---

## 6. ruff check

```
All checks passed!
```

---

## 7. 关键设计决策备注

| 决策 | 理由 |
|---|---|
| 3 个 prompt 函数均为纯函数（无 I/O） | 可并发调用、易测试、无副作用；provider 注入留给 first_last_frame_transition / video_edit_element_remove |
| intensity `[0,0.33]→slow / (0.33,0.67]→medium / (0.67,1]→fast` | 三段等宽映射，边界 0.33/0.67 落入低段（含端点） |
| `LLMCaller` Protocol（非 ABC） | 允许鸭子类型：任何接受 `messages=` kwarg 并返回 `dict` 的 callable 均可，无需继承 |
| `story_predict` 同步调用 LLM | LLM 调用往往是同步阻塞（各 SDK 的 `.chat()` 方法），包装为 async 函数但内部 `llm(messages=...)` 同步 |
| `invoke_cloud` TECHNICAL_DEBT stub | 2026-05-27 Meituan 无官方 LongCat 云 API；第三方 fal.ai/WaveSpeed 非官方；留 stub 文档化现状 |
| `FrameTransitionError` / `VideoEditError` 层级 | provider 抛出的所有未知异常统一包装，防止 I/O 错误泄漏到调用方 |

---

## 8. 边界确认

- 修改范围：`/home/soffy/projects/platform/oprim/` 仅
- 不涉及 hevi / obase / stratum
- `longcat_avatar` 不在顶层 `__init__.py.__all__`（私有 vendor wrapper）
