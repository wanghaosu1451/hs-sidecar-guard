# Skill: python-tester
描述：为任一段 Python 模块自动生成并运行单元测试，输出靠 pytest 校验的结果。
玩法：1) 读入目标代码；2) 生成 test_*.py；3) 用 run_shell 跑 pytest -q；4) 修复失败到通过。
