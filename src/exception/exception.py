import sys

class customException(Exception):
    
    def __init__(self, error_message, error_details: sys):
        self.errorMsg = error_message
        _,_,exc_info = error_details.exc_info()
        
        self.lineNumber = exc_info.tb_lineno
        self.fileName = exc_info.tb_frame.f_code.co_filename
                          
    def __str__(self):
        return "Error occured in named python folder [{0}] and line number [{1}]. The error was [{2}]".format(
            self.fileName, self.lineNumber, self.errorMsg
        )