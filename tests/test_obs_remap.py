from srl.utils.obs_remap import apply_obs_remap


def test_apply_obs_remap_with_explicit_input_names() -> None:
    remapped = apply_obs_remap(
        {"camera": "img", "joint_states": "vec"},
        ["visual_enc", "state_enc"],
        {"visual_enc": "camera", "state_enc": "joint_states"},
    )

    assert remapped == {"visual_enc": "img", "state_enc": "vec"}


def test_apply_obs_remap_single_obs_single_encoder() -> None:
    remapped = apply_obs_remap({"policy": 123}, ["policy_enc"], {})
    assert remapped == {"policy_enc": 123}