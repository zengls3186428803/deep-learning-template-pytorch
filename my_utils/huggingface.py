import os


class HF_MIRROR:
    HF_ENDPOINT = None

    @staticmethod
    def set_hf_mirror(server="https://hf-mirror.com"):
        import os
        os.environ["HF_ENDPOINT"] = server
        HF_MIRROR.HF_ENDPOINT = os.environ.get("HF_ENDPOINT")

    @staticmethod
    def unset_hf_mirror(clear=False):
        if not clear:
            os.environ["HF_ENDPOINT"] = HF_MIRROR.HF_ENDPOINT
        else:
            os.environ["HF_ENDPOINT"] = ""
