# Loads raw documents from disk (PDF, TXT)
import pymupdf

class FileLoader:
    def __init__(self, filepath):
        self.filepath = filepath

    def get_file_type(self):
        filetype = self.filepath.split(".")[-1]
        filename = self.filepath.split("/")[-1]
        if filetype == 'pdf':
            return [filename, "PDF"]
        elif filetype == 'txt':
            return [filename,"TEXT"]
        else:
            return [filename, "NotDefined"]

    def read_file(self):
        self.filename, self.filetype = self.get_file_type()[0], self.get_file_type()[1]
        self.file_metadata = {}
        self.file_metadata['file_name'] = self.filename
        if self.filetype == 'PDF':
            doc = pymupdf.open(self.filepath)
            texts = ''
            for page in doc:
                texts += page.get_text()
            self.file_metadata['file_content'] = texts
        
        elif self.filetype == 'TEXT':
            texts = ''
            with open(self.filepath, "r") as textfile:
                texts = textfile.read()
            textfile.close()
            self.file_metadata['file_content'] = texts

        else:
            self.file_metadata['file_content'] = None
        # need to figure out if page number should be included in metadata
        # if yes, how? (especially in txt files)
        return self.file_metadata
            
