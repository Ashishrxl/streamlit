import streamlit as st 






st.title("Hello, BharatGPT!") st.write("This is your first Streamlit app using BharatGPT Mini.")



user_prompt = st.text_input("Enter your prompt:", "Explain generative AI in one sentence.")

Add a slider for temperature

temperature = st.slider( "Model temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.01, help="Controls randomness: 0 = deterministic, 1 = very creative" )

with st.spinner("AI is working..."):  st.write('Hello')



tool_script = """

<script type="text/javascript">
    var s = document.createElement("script");
    s.src = "https://builder.corover.ai/params/widget/corovercb.lib.min.js?appId=d977ea27-874f-411c-a08e-cc352dcdf0f2";
    s.type = "text/javascript";
    document.getElementsByTagName("body")[0].appendChild(s);
</script>"""



st.components.v1.html(tool_script, height=800, scrolling=True)


