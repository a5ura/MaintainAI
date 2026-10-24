"""Streamlit entry point. Run: python -m streamlit run app.py."""

import sqlite3

import streamlit as st

from maintainai.database import connect, default_db_path, initialize
from maintainai.service import MaintenanceStore
from maintainai.ui import render


def main() -> None:
    st.set_page_config(page_title="MaintainAI", page_icon="🔧", layout="wide")
    st.title("MaintainAI")
    st.caption("Industrial equipment maintenance • local SQLite workspace")
    st.info("Demo data notice: all bundled assets, incidents, people, findings, and actions are synthetic. Brand names identify example equipment only.")
    try:
        with connect() as connection:
            initialize(connection)
            render(MaintenanceStore(connection))
    except sqlite3.Error:
        st.error("The SQLite database could not be read or written. Check that its folder is writable and another process is not holding a long-running transaction.")
        st.code(str(default_db_path()))


if __name__ == "__main__":
    main()
