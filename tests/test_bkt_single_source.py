"""D5 单源守卫：oprim.bkt 的无前缀名必须就是 oprim._cognitive 的实现本体。

防止未来有人在 bkt.py 里重新 fork 一份 BKT 算法（"改一份漏另一份"）。
"""
import oprim.bkt as B
import oprim._cognitive as Canon   # 唯一事实来源
import oprim.cognitive as Pub      # 公开 re-export


def test_bkt_is_single_source():
    # bkt.py 无前缀名 == canonical 实现本体
    assert B.bkt_update is Canon.bkt_update
    assert B.classify_error is Canon.bkt_classify_error
    assert B.predict_correct is Canon.bkt_predict_correct
    assert B.new_state_from_prior is Canon.bkt_new_state
    assert B.exp_forgetting is Canon.exp_forgetting
    assert B._item_adjust is Canon._item_adjust
    # 公开 re-export 也指向同一本体
    assert Pub.bkt_update is Canon.bkt_update


def test_no_inline_algorithm_in_bkt_module():
    """bkt.py 不应再自带算法函数定义（只允许 re-export 别名）。"""
    import inspect
    import oprim.bkt as mod
    src = inspect.getsource(mod)
    # 别名层里不出现贝叶斯更新的关键算式（避免重新内联实现）
    assert "p_obs" not in src and "P_L_obs" not in src
