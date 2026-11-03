"""Streamlit 入口：先渲染全局风险横幅，再路由到四个页面。"""

import streamlit as st

from lottery.ui.components import render_disclaimer_banner


def route() -> None:
    """注册四个页面并运行当前页。"""
    pages = [
        st.Page("pages/1_teaching.py", title="教学演示", icon="📚"),
        st.Page("pages/2_math.py", title="数学最优", icon="🧮"),
        st.Page("pages/3_data.py", title="数据管理", icon="🗂️"),
        st.Page("pages/4_randomness.py", title="随机性检验", icon="🎲"),
    ]
    st.navigation(pages).run()


def main() -> None:
    """配置页面，并保证横幅先于任何页面内容出现。"""
    st.set_page_config(page_title="双色球教学演示系统", layout="wide")
    render_disclaimer_banner()
    route()


if __name__ == "__main__":
    main()
