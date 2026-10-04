"""Pytest configuration.

Ép toàn bộ unit tests chạy OFFLINE (không gọi LLM/API) để nhanh, ổn định và
không phụ thuộc mạng. Pipeline thật (`python main.py`) vẫn dùng LLM bình thường.
"""

import os

os.environ["LAB18_OFFLINE"] = "1"
