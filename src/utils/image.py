import os
import magic
import requests
import base64


class Image:
    """Image Utils"""

    @staticmethod
    def encode_image(image_path_or_url: str, get_mime_type: bool = False):
        """Encode image into base64 with utf-8"""
        if image_path_or_url.startswith("http"):
            try:
                response = requests.get(image_path_or_url, stream=True)
                response.raise_for_status()
                image = response.content
                mime_type = response.headers.get("content-type", None)
                base64_encoded = base64.b64encode(image).decode('utf-8')
                if get_mime_type:
                    return base64_encoded, mime_type
                else:
                    return base64_encoded
            except requests.exceptions.RequestException as e:
                print(f"Request error: {e}")
                if get_mime_type:
                    return None, None

                return None
        else:
            if not os.path.exists(image_path_or_url):
                return None, None
            mime_type = magic.Magic(mime=True).from_file(image_path_or_url)
            if mime_type.startswith("image/"):
                with open(image_path_or_url, "rb") as image_file:
                    if get_mime_type:
                        return base64.b64encode(
                            image_file.read()
                        ).decode("utf-8"), mime_type
                    
                    return base64.b64encode(image_file.read()).decode("utf-8")
            else:
                if get_mime_type:
                    return None, None
                
                return None
