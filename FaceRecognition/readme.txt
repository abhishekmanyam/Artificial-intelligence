face recognition with age and gender classification

1.keep your images in ids directory
eg:
      ids
       |
       name1
          |
           pic1.jpg
           .
           .
           . 
           picn.jpg
       |
       name2
          |
           pic1.jpg
           .
           .
           . 
           picn.jpg


3. Make sure where your model is placed(.pb file).

4. To start the program, use ("python main.py modelpath idpath"):"python main.py ./model/xxxx.pb ./ids"
5. While the program is running..
   press f for fps
   press s to load the images
   press t to train the images(this will stop the detecting process)
   press q to exit process

6.Make sure atleast one image is placed in directory.

7.To change the limit of no of persons that can be trained, change the count value in the test function function(file:main.py) default set to 10