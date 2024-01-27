import io
import joblib
from exasol.bucketfs import (
    Service,
    as_bytes,
    as_file,
    as_string,
)

class bucketfs_client():
    """
        @brief Provide client API to interact with BucketFS
    """
    url = "http://172.18.0.2:6583" # Change this to your script language container IP
    cred = {"default":{"username":"w","password":"write"}}
    is_available = True
    
    @classmethod
    def init(cls):
        try:
            cls.bucketfs = Service(cls.url, cls.cred)
            cls.bucket = cls.bucketfs["default"]
        except Exception as e:
            print(f"Warning: Connection to bucketfs_client {cls.url} under user {cls.cred['default']['username']} failed")
            cls.is_available = False
    
    @classmethod
    def upload(cls,file_path,buffer):
        """
         @brief Upload file to BucketFS
         @param file_path: Full path and file name in BucketFS
         @parm buffer: Buffer object containing data to upload
        """       
        cls.init()
        buffer.seek(0)             
        cls.bucket.upload(file_path, buffer)

    @classmethod
    def download(cls,file_path):
        """
         @brief Download file to BucketFS
         @param file_path: Full path and file name in BucketFS
        """
        cls.init()
        data = io.BytesIO(as_bytes(cls.bucket.download(file_path)))
        return data

    @classmethod
    def check(cls,file_path):
        """
         @brief Check if file is in BucketFS
         @param file_path: Full path and file name in BucketFS
        """
        cls.init()
        for file in cls.bucket:
            if (file == file_path):
                return True
        return False

    @classmethod
    def delete(cls,file_path):
        """
         @brief Delete file in BucketFS
         @param file_path: Full path and file name in BucketFS
        """
        cls.init()
        cls.bucket.delete(file_path)

    @classmethod
    def view(cls):
        """
         @brief View all files in BucketFS
        """
        cls.init()
        for file in cls.bucket:
            print(file)