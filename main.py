from ImageProcessing import Processing

if __name__ == "__main__":
    x = Processing()
    Imagepath = r'C:\Users\Jonas\OneDrive\Desktop\Python\Image.jpg'
    Savepath = r'C:\Users\Jonas\OneDrive\Desktop\Python'
    x.SaveImage(Imagepath, Savepath)
    x.ShowImage(Imagepath)