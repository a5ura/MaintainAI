"""Local-only sidebar status; cloud configuration is not cloud authorization."""

import streamlit as st

from maintainai.llm import get_llm_config_status


def cloud_configuration_status() -> None:
    st.subheader("Cloud AI Configuration")
    status = get_llm_config_status()
    if status.status == "Configured":
        st.success("Configured")
        st.text("Model: " + status.model)
    elif status.status == "Not configured":
        st.info("Not configured")
        st.text("Missing: " + ", ".join(status.missing_variables))
    else:
        st.warning("Invalid configuration")
    st.caption(status.message)
    st.caption("Local validation only; no provider request is made. Cloud generation still requires explicit consent.")
