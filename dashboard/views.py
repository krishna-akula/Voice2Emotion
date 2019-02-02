from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .codes.ml_utils import dir_predict


def index(request):
    if request.method == 'POST' and request.FILES['myfile']:
        myfile = request.FILES['myfile']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        ans = dir_predict(fs.url(filename))
        return render(request, 'index.html', {
            'uploaded_file_url': uploaded_file_url, 
            'result' : ans
        })
    return render(request, 'index.html')