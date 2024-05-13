import io
import joblib
from exasol.bucketfs import (
    Service,
    as_bytes,
    as_file,
    as_string,
)

class BucketFSClient:
    """
        @brief Provide client API to interact with BucketFS
    """    
    @classmethod
    def __init__(self, url: str, bucket: str, user: str, password: str):
        self.url = url
        self.cred = {bucket: {"username": user, "password": password}}
        try:            
            self.bucketfs = Service(self.url, self.cred)
            self.bucket = self.bucketfs["default"]
        except Exception as e:
            print(f"Warning: Connection to bucketfs_client {self.url} under user {self.cred['default']['username']} failed")            
    
    @classmethod
    def upload(self,file_path,buffer):
        """
         @brief Upload file to BucketFS
         @param file_path: Full path and file name in BucketFS
         @parm buffer: Buffer object containing data to upload
        """       
        buffer.seek(0)             
        self.bucket.upload(file_path, buffer)

    @classmethod
    def download(self,file_path):
        """
         @brief Download file to BucketFS
         @param file_path: Full path and file name in BucketFS
        """
        data = io.BytesIO(as_bytes(self.bucket.download(file_path)))
        return data

    @classmethod
    def check(self,file_path):
        """
         @brief Check if file is in BucketFS
         @param file_path: Full path and file name in BucketFS
        """
        for file in self.bucket:
            if (file == file_path):
                return True
        return False

    @classmethod
    def delete(self,file_path):
        """
         @brief Delete file in BucketFS
         @param file_path: Full path and file name in BucketFS
        """
        self.bucket.delete(file_path)

    @classmethod
    def view(self):
        """
         @brief View all files in BucketFS
        """
        for file in self.bucket:
            print(file)