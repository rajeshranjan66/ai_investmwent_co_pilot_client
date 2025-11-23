
import random
import streamlit as st


def generate_math_captcha(location_key: str):
    """Generate or reuse a simple math question in session state with +, -, *."""
    a_key = f"captcha_a_{location_key}"
    b_key = f"captcha_b_{location_key}"
    op_key = f"captcha_op_{location_key}"

    if a_key not in st.session_state or b_key not in st.session_state or op_key not in st.session_state:
        st.session_state[a_key] = random.randint(1, 9)
        st.session_state[b_key] = random.randint(1, 9)
        st.session_state[op_key] = random.choice(['+', '-', '*'])

    a = st.session_state[a_key]
    b = st.session_state[b_key]
    op = st.session_state[op_key]

    if op == '+':
        answer = a + b
    elif op == '-':
        answer = a - b
    else:
        answer = a * b

    return f"{a} {op} {b} = ?", answer


def validate_math_captcha(user_input: str, location_key: str) -> bool:
    """Check if the answer is correct for +, -, *."""
    try:
        a_key = f"captcha_a_{location_key}"
        b_key = f"captcha_b_{location_key}"
        op_key = f"captcha_op_{location_key}"
        a = st.session_state[a_key]
        b = st.session_state[b_key]
        op = st.session_state[op_key]
        if op == '+':
            expected_answer = a + b
        elif op == '-':
            expected_answer = a - b
        else:
            expected_answer = a * b
        return int(user_input.strip()) == expected_answer
    except:
        return False





def reset_captcha(location_key: str):
    """Reset captcha for the given location and regenerate a new one"""
    keys_to_remove = [
        f"captcha_question_{location_key}",
        f"captcha_verified_{location_key}",
        f"captcha_input_{location_key}",
        f"captcha_a_{location_key}",
        f"captcha_b_{location_key}",
        f"captcha_op_{location_key}"
    ]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]

    question, correct_answer = generate_math_captcha(location_key)
    st.session_state[f"captcha_question_{location_key}"] = question


def render_captcha(location_key: str) -> bool:
    """Render captcha input field and return if it's verified"""
    verified_key = f"captcha_verified_{location_key}"
    error_key = f"captcha_error_{location_key}"

    # Generate or refresh the CAPTCHA
    if f"captcha_question_{location_key}" not in st.session_state:
        reset_captcha(location_key)

    # Display the current CAPTCHA question
    st.markdown(f"**Captcha**: Solve → `{st.session_state[f'captcha_question_{location_key}']}`")
    user_answer = st.text_input("Enter the answer (just the number):", key=f"captcha_input_{location_key}")

    if user_answer:
        if validate_math_captcha(user_answer, location_key):
            st.session_state[verified_key] = True
        else:
            st.session_state[verified_key] = False
            st.session_state[error_key] = "Please complete the CAPTCHA correctly"
            reset_captcha(location_key)

    # Display error message if present
    if error_key in st.session_state:
        st.error(st.session_state[error_key])
        del st.session_state[error_key]
        st.rerun()  # Trigger rerun after showing the error

    return st.session_state.get(verified_key, False)
