import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        return frame

st.title("Streamlit WebRTC Example")

# Stream the webcam feed using our VideoTransformer
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
