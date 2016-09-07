import requests

# from http://stackoverflow.com/a/16696317
def download_file(url, filename):
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True, allow_redirects=True)
    with open(filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                #f.flush() commented by recommendation from J.F.Sebastian
    return filename


download_file(
    "https://www.dropbox.com/sh/sk74asltc1rwqnf/AAAGG_sbzjxcJTKlix3DWNkQa?dl=1",
    "pk-EFM.zip")