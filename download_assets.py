import shutil

import requests
import zipfile


def download(url, save_path, chunk_size=128):
	r = requests.get(url, stream=True)
	with open(save_path, 'wb') as fd:
		for chunk in r.iter_content(chunk_size=chunk_size):
			fd.write(chunk)


def unzip():
	local_zip = 'computer-vision-assets.zip'
	zip_ref = zipfile.ZipFile(local_zip, 'r')
	zip_ref.extractall('computer-vision-assets-extracted')
	zip_ref.close()


if __name__ == "__main__":
	print("Downloading zip...")
	url = 'http://apssouza.com.br/downloads/computer-vision-assets.zip'
	download(url, "computer-vision-assets.zip")
	print("Unzipping...")
	unzip()
	shutil.move("computer-vision-assets-extracted/downloads", "downloads")
	shutil.rmtree("computer-vision-assets-extracted")
	print("The assets was download successfully")
