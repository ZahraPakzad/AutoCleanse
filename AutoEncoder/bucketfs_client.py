import io
from exasol.bucketfs import Service
from exasol.bucketfs import (
    Service,
    as_bytes,
    as_file,
    as_string,
)

class BucketFS_client():
    def __init__(self):
        """
         @brief Provide client API to interact with BucketFS
        """
        self.url = "http://172.18.0.2:6583"
        self.cred = {"default":{"username":"w","password":"write"}}
        self.bucketfs = Service(self.url,self.cred)
        self.bucket = self.bucketfs["default"]  
    
    def upload(self,file_path,buffer):
        """
         @brief Upload file to BucketFS
         @param file_path: Full path and file name in BucketFS
         @param buffer: BytesIO object to write to BucketFS as bytes
        """
        buffer.seek(0)             
        self.bucket.upload(file_path, buffer)

    def download(self,file_path):
        """
         @brief Download file to BucketFS
         @param file_path: Full path and file name in BucketFS
        """
        data = data = io.BytesIO(as_bytes(self.bucket.download(file_path)))
        return data