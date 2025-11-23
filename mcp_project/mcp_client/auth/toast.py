import streamlit as st
#
# def show_toast(message, type="success", duration=3000):
#     color = "#4BB543" if type == "success" else "#FF4B4B"
#     icon = "✅" if type == "success" else "❌"
#     st.markdown(f"""
#     <div id="toast" style="
#         position: fixed;
#         top: 30px;
#         right: 30px;
#         z-index: 9999;
#         background: {color};
#         color: white;
#         padding: 16px 28px;
#         border-radius: 8px;
#         font-size: 16px;
#         box-shadow: 0 2px 8px rgba(0,0,0,0.15);
#         display: flex;
#         align-items: center;
#         gap: 10px;
#         animation: fadeIn 0.5s;
#     ">
#         <span>{icon}</span>
#         <span>{message}</span>
#     </div>
#     <script>
#     setTimeout(function(){{
#         var toast = document.getElementById('toast');
#         if (toast) toast.style.display = 'none';
#     }}, {duration});
#     </script>
#     <style>
#     @keyframes fadeIn {{
#         from {{ opacity: 0; transform: translateY(-20px); }}
#         to {{ opacity: 1; transform: translateY(0); }}
#     }}
#     </style>
#     """, unsafe_allow_html=True)


import streamlit as st

import streamlit as st


def show_toast(message, type="success", duration=3000, position="top"):
    """
    Display a toast notification

    Args:
        message: The message to display
        type: "success" or "error"
        duration: Duration in milliseconds (default: 3000)
        position: "top", "bottom", or "middle"
    """
    color = "#4BB543" if type == "success" else "#FF4B4B"
    icon = "✅" if type == "success" else "❌"

    # Set position CSS based on parameter
    if position == "bottom":
        position_css = "bottom: 30px; top: auto;"
        animation = "fadeInBottom 0.5s;"
    elif position == "middle":
        position_css = "top: 50%; transform: translateY(-50%);"
        animation = "fadeInMiddle 0.5s;"
    else:  # default to top
        position_css = "top: 30px; bottom: auto;"
        animation = "fadeIn 0.5s;"

    st.markdown(f"""
    <div id="toast" style="
        position: fixed;
        {position_css}
        right: 30px;
        z-index: 9999;
        background: {color};
        color: white;
        padding: 16px 28px;
        border-radius: 8px;
        font-size: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        display: flex;
        align-items: center;
        gap: 10px;
        {animation}
    ">
        <span>{icon}</span>
        <span>{message}</span>
    </div>
    <script>
    setTimeout(function(){{
        var toast = document.getElementById('toast');
        if (toast) toast.style.display = 'none';
    }}, {duration});
    </script>
    <style>
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(-20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes fadeInBottom {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes fadeInMiddle {{
        from {{ opacity: 0; transform: translateY(-20px) scale(0.9); }}
        to {{ opacity: 1; transform: translateY(-50%) scale(1); }}
    }}
    </style>
    """, unsafe_allow_html=True)