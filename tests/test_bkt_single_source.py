"""D5 单源守卫：oprim.bkt 的无前缀名必须就是 oprim._cognitive 的实现本体。

防止未来有人在 bkt.py 里重新 fork 一份 BKT 算法（"改一份漏另一份"）。
"""

import oprim._cognitive as canonical  # 唯一事实来源
import oprim.bkt as bkt_mod
import oprim.cognitive as public  # 公开 re-export


def test_bkt_is_single_source():
    # bkt.py 无前缀名 == canonical 实现本体
    assert bkt_mod.bkt_update is canonical.bkt_update
    assert bkt_mod.classify_error is canonical.bkt_classify_error
    assert bkt_mod.predict_correct is canonical.bkt_predict_correct
    assert bkt_mod.new_state_from_prior is canonical.bkt_new_state
    assert bkt_mod.exp_forgetting is canonical.exp_forgetting
    assert bkt_mod._item_adjust is canonical._item_adjust
    # 公开 re-export 也指向同一本体
    assert public.bkt_update is canonical.bkt_update


def test_no_inline_algorithm_in_bkt_module():
    """bkt.py 不应再自带算法函数定义（只允许 re-export 别名）。"""
    import inspect

    import oprim.bkt as mod

    src = inspect.getsource(mod)
    # 别名层里不出现贝叶斯更新的关键算式（避免重新内联实现）
    assert "p_obs" not in src and "P_L_obs" not in src
