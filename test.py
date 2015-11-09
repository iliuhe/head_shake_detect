# -*- coding: utf-8 -*-
"""
Created on Thu Oct 01 19:58:15 2015

@author: He
"""

import cv2
#import sys
import numpy as np

import winsound
import os
from PIL import Image,ImageTk
import Tkinter


#import matplotlib.pyplot as plt

class simpleapp_tk(Tkinter.Tk):
    image_type=['PNG','JPG','JP2','BMP','TIF','GIF']
    file_names=[]
    cur_image_idx=0
    state_eye,state_hand=0,0
    eye_xy,hand_xy=[],[]
    #pre_frame,flow=[],[]
    def plot_graph(hist_item):
        h = np.zeros((300,256,3))
        #hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
        cv2.normalize(hist_item,hist_item,0,255,cv2.NORM_MINMAX)
        hist=np.int32(np.around(hist_item))
        for x,y in enumerate(hist):
            cv2.line(h,(x,0),(x,y),(255,255,255))
        y = np.flipud(h)
        return y
    
    
    def webcam(self,frame,eye=1,hand=1,opt_flow=1,show=0):
        #cascPath = sys.argv[1]
        #Cascade = cv2.CascadeClassifier(cascPath1)
        #temp=None    
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        fgmask = self.fgbg.apply(gray)
            
        if hand and self.state_eye==0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))    
            
            # erode and dilate
            kernel=np.ones((11,11),np.uint8)
            d1=cv2.dilate(fgmask,kernel,iterations=1)
            e1=cv2.erode(d1,kernel,iterations=1)
            d2=cv2.dilate(e1,kernel,iterations=1)
            e2=cv2.erode(d2,kernel,iterations=1)
            fgmask = cv2.morphologyEx(e2, cv2.MORPH_OPEN, kernel)
            #ret,thresh=cv2.threshold(e2,180,255,0)  
            #cv2.findContours()
            im2, contours, hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            selected_contours=[]
            for i in range(len(contours)):
                selected_contours.append(cv2.contourArea(contours[i]))
            if selected_contours!=[] and max(selected_contours)>50000:
                #print 'find!'
                max_contour=contours[selected_contours.index(max(selected_contours))]  
                #print len(max_contour)
                
                #cv2.drawContours(fgmask, max_contour, -1, (255,255,255), 3)
                
                
                if len(self.hand_xy)>50:
                    self.hand_xy.pop(0)
                
                #(x_c,_),radius = cv2.minEnclosingCircle(max_contour)
                
#                rect=cv2.minAreaRect(max_contour)
#                box=cv2.boxPoints(rect)
                
                #print cv2.convexHull(max_contour)
                
                x,_,w,_=cv2.boundingRect(max_contour)
                #self.hand_xy.append((x+w/2+sum(box[0])/2)/2)
                
                self.hand_xy.append(x+w/2)
                #self.hand_xy.append(sum(box[0]/2))
    
                #hand_xy.append(np.median([max_contour[i][0][0] for i in range(len(max_contour))]))
            else:
                if len(self.hand_xy)>5:
                    temp_xy=[self.hand_xy[-3],self.hand_xy[-2],self.hand_xy[-1]]
                    dif1=temp_xy[1]-temp_xy[0]
                    dif2=temp_xy[2]-temp_xy[1]
                    if self.state_hand==0 and not (0 in temp_xy):
                        print temp_xy,dif1,dif2
                        if abs(dif1)>abs(dif2):
                            dif=dif1
                        else:
                            dif=dif2
                        if dif>30:# right
                            print 'right'
                            self.Button_next()
                            winsound.Beep(3000,100)
                            self.state_hand=-1
                            self.hand_xy.append(0)
                            return
                        elif dif<-30:# left
                            print 'left'
                            self.Button_pre()
                            winsound.Beep(4000,100)
                            self.state_hand=1
                            self.hand_xy.append(0)
                            return
                    elif sum(self.hand_xy[-6:-1])==0:
                        self.state_hand=0
                self.hand_xy.append(0)
            #print np.shape(max_contour)

#        for i in range(len(max_contour)):
#            for j in range(len(max_contour[i])):
#                x.append(max_contour[i][j][0])
#        print x
        
        
        
            #hist_item2 = cv2.calcHist([fgmask],[0],None,[256],[0,256])
            #cv2.imshow('histogram2',self.plot_graph(hist_item2))        
            
#                s=sum(fgmask)
#                #print s
#                h = np.zeros((300,256,3))
#                #hist_item = cv2.calcHist([im],[0],None,[256],[0,256])
#                cv2.normalize(s,s,0,255,cv2.NORM_MINMAX)
#                hist=np.int32(np.around(s))
#                for x,y in enumerate(hist):
#                    cv2.line(h,(x,0),(x,y),(255,255,255))
#                y = np.flipud(h)
            
            #cv2.imshow('dif',y)
        
#        if opt_flow and self.state_eye==0 and self.state_hand==0 and self.pre_frame!=[]:
#            self.flow = cv2.calcOpticalFlowFarneback(self.pre_frame,gray,None,0.5,3,15,1,5,1.2,0)
#            #print np.shape(self.flow)
#            temp=sum(sum(self.flow[:,:,1]))
#            self.l.append(sum(sum(self.flow[:,0:200,1])))
#            self.m.append(sum(sum(self.flow[:,200:480,1])))
#            self.r.append(sum(sum(self.flow[:,480:680,1])))
#
#            if len(self.head_xy)>5:
#                self.head_xy.pop(0)
#            if abs(temp)<50000:
#                self.head_xy.append(0)
#            else:
#                self.head_xy.append(temp)
#            print self.head_xy
#            if self.state_head==0:
#                if self.head_xy[-1]>150000 and self.head_xy[-3:-1]==[0,0]:
#                    self.state_head=-1
#                    print 'next'
#                    self.Button_next()
#                    winsound.Beep(3000,100)
#                elif self.head_xy[-1]<-150000 and self.head_xy[-3:-1]==[0,0]:
#                    self.state_head=1
#                    print 'pre'
#                    self.Button_pre()
#                    winsound.Beep(4000,100)
#            elif self.head_xy==[0]*6:
#                self.state_head=0
#                    
#            
#            
        
        
        '''
        situation below is for detecting the head movement by rotate head
        '''
        if eye and self.state_hand==0:
            eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')
            dectet_eye = eyeCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            # find two eyes
            if len(dectet_eye)>2:
                temp=[dectet_eye[i] for i in range(len(dectet_eye)) if 150<(dectet_eye[i][1]+dectet_eye[i][3]/2)<350 and 200<(dectet_eye[i][0]+dectet_eye[i][2]/2)<480]
                dectet_eye=temp
            for (x, y, w, z) in dectet_eye:
                    cv2.rectangle(gray, (x, y), (x+w, y+z), (0, 255, 0), 2)
            if len(dectet_eye)==2 and abs(dectet_eye[0][1]+dectet_eye[0][3]/2-dectet_eye[1][1]-dectet_eye[1][3]/2)<30:
                print 'find two eyes'
                #print 'find two eyes, len(xy)='+str(len(xy))
                mid=np.average([dectet_eye[0][0]+dectet_eye[0][2]/2,dectet_eye[1][0]+dectet_eye[1][2]/2])
                print mid,self.state_eye
                self.eye_xy.append(mid)
                if len(self.eye_xy)>100:
                    self.eye_xy.pop(0)
                if len(self.eye_xy)>5:
                    latest5=self.eye_xy[-6:-1]
                    #print latest5
                    #print np.var(np.array(latest5))
                    if self.state_eye==0:
                        latest3=np.append(latest5[2:3],latest5[3])
                        latest2=np.append(latest5[1:2],latest5[2])
                        dif=sum(latest3-latest2)
                        print dif
                        #if dif<-20:# sensitivity of turn left and right, smaller number gives more sensitive
                        if self.eye_xy[-1]-self.eye_xy[-2]<-20:# and self.eye_xy[-2]-self.eye_xy[-3]<-6:
                            self.Button_pre()
                            winsound.Beep(4000,100)
                            self.state_eye=-1
                            return
                        elif self.eye_xy[-1]-self.eye_xy[-2]>20:# and self.eye_xy[-2]-self.eye_xy[-3]>6:
                            self.Button_next()
                            winsound.Beep(3000,100)
                            self.state_eye=1
                            return
                    else:
                        if np.var(latest5)<3:# adjust stable time, smaller number gives longer stablize time
                            self.state_eye=0
                
                
            # does not find two eyes

                #xy.append(np.array([[0,0,0,0],[0,0,0,0]]))
        #self.pre_frame=gray
        if show:
            #hist_item = cv2.calcHist([gray],[0],None,[256],[0,256])
            #cv2.imshow('histogram',self.plot_graph(hist_item))
            if self.tiker1_str.get()==1:
                cv2.imshow('Eye Matching', gray)
            if self.tiker2_str.get()==1:
                cv2.imshow('Hand detect',fgmask)
#            if self.tiker3_str.get()==1 and self.flow!=[]:
#                cv2.imshow('Optical flow',self.draw_flow(gray,self.flow))
    
                
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
        #hf.list2txt('F:/xy.txt',xy)
        #print hand_xy
    def draw_flow(self,im,flow,step=16):
        h,w = im.shape[:2]
        y,x = np.mgrid[step/2:h:step,step/2:w:step].reshape(2,-1)
        fx,fy = flow[y,x].T
    
        # create line endpoints
        lines = np.vstack([x,y,x+fx,y+fy]).T.reshape(-1,2,2)
        lines = np.int32(lines)
    
        # create image and draw
        vis = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        for (x1,y1),(x2,y2) in lines:
            cv2.line(vis,(x1,y1),(x2,y2),(0,255,0),1)
            cv2.circle(vis,(x1,y1),1,(0,255,0), -1)
        return vis
    
    def load_image(self,location):
        if not os.path.exists(location):
            return
        for file_name in os.listdir(location):
            if file_name.split('.')[-1].upper() in self.image_type:
                self.file_names.append(location+'/'+file_name)
        self.cur_image_idx=0
    
    def image_resize(self,dirr):
        img=Image.open(dirr)
        img.thumbnail([512,512],Image.ANTIALIAS)
        return img
        
        #if location:
            #self.load_image(location)
            #cv2.imread(self.file_names[0])
            #self.cur_image_idx=0
        #self.webcam('haarcascade_eye.xml',eye=1,hand=1)
            
    def show_frame(self):
        #print 'cam_on'+str(self.cam_on)
        if self.cam_on==0:
            return
        if self.state_eye==0 and self.state_hand==0 :
            self.recog_str.set('Ready')
            self.recog.configure(background='blue')
        if self.pre_tiker1!=self.tiker1_str.get():
            self.state_eye=0
            self.eye_xy=[]
            self.pre_tiker1=self.tiker1_str.get()
            if self.tiker4_str.get()==1:
                cv2.destroyWindow('Eye Matching')
        if self.pre_tiker2!=self.tiker2_str.get():
            self.state_hand=0            
            self.hand_xy=[]
            self.pre_tiker2=self.tiker2_str.get()
            if self.tiker4_str.get()==1:
                cv2.destroyWindow('Hand detect')
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.webcam(cv2image,eye=self.tiker1_str.get(),hand=self.tiker2_str.get(),show=self.tiker4_str.get())
        self.cam.imgtk = imgtk
        self.cam.configure(image=imgtk)
        if self.tiker4_str.get()==0:
            cv2.destroyAllWindows()
        self.cam.after(5, self.show_frame)
################################################################
# GUI
        
    def __init__(self,parent):
        Tkinter.Tk.__init__(self,parent)
        self.parent = parent
        self.initialize()

    def initialize(self):
        
        self.grid()
        
        img_idx_str=Tkinter.Label(self,width=20,text=u'Image folder dir: ')   
        img_idx_str.grid(column=0,row=0,sticky='EW')

        
        self.entryVariable=Tkinter.StringVar()
        self.entry=Tkinter.Entry(self,textvariable=self.entryVariable)
        self.entry.grid(column=1,row=0,sticky='EW')
        self.entry.bind("<Return>",self.OnPressEnter)
        #self.entryVariable.set(u'Enter text here.')
        self.entryVariable.set(u'ref_orig')
        
        self.recog_str=Tkinter.StringVar()
        self.recog_str.set('None')
        self.recog=Tkinter.Label(self,fg="white",bg='blue',textvariable=self.recog_str)
        self.recog.grid(column=4,row=2,columnspan=3)
        
        button=Tkinter.Button(self,text=u"Load Images",command=self.OnButtonClick)
        button.grid(column=2,row=0)
        
        button_next=Tkinter.Button(self,text=u"next",command=self.Button_next)
        button_next.grid(column=2,row=1)
        
        button_pre=Tkinter.Button(self,text=u"previous",command=self.Button_pre)
        button_pre.grid(column=0,row=1)
        
        
        self.img_idx_str=Tkinter.StringVar()
        #img_idx_str.set('Hello')
        label = Tkinter.Label(self,anchor="w",fg="white",bg="blue",textvariable=self.img_idx_str)
        label.grid(column=1,row=1)
        self.img_idx_str.set(u'No image loaded')
        
        
        self.grid_columnconfigure(0,minsize=5)
        self.resizable(False,False)
        
        self.image=ImageTk.PhotoImage(self.image_resize('no_image_loaded.png'))
        self.label2=Tkinter.Label(self,image=self.image)
        self.label2.grid(column=0,row=3,columnspan=3)
        
        self.cam_img=ImageTk.PhotoImage(Image.open('camera.png'))
        self.cam=Tkinter.Label(self,image=self.cam_img)
        self.cam.grid(column=4,row=3,columnspan=3)
        
        button_open_cam=Tkinter.Button(self,text=u'Open Camera',command=self.Button_open_cam)
        button_open_cam.grid(column=4,row=0)
        
        button_close_cam=Tkinter.Button(self,text=u'Close Camera',command=self.Button_close_cam)
        button_close_cam.grid(column=5,row=0)
        
        button_exit=Tkinter.Button(self,text=u'Exit',command=self.Button_exit)
        button_exit.grid(column=6,row=0)
        
        self.tiker1_str=Tkinter.IntVar()
        tiker1=Tkinter.Checkbutton(self,text="head turn detect",variable=self.tiker1_str)
        tiker1.grid(column=4,row=1)
        
        self.tiker2_str=Tkinter.IntVar()
        tiker2=Tkinter.Checkbutton(self,text="hand wave detect",variable=self.tiker2_str)
        tiker2.grid(column=5,row=1)
        
#        self.tiker3_str=Tkinter.IntVar()
#        tinker3=Tkinter.Checkbutton(self,text='head swing detect',variable=self.tiker3_str)  
#        tinker3.grid(column=6,row=1)
        
        self.tiker4_str=Tkinter.IntVar()
        tiker4=Tkinter.Checkbutton(self,text='show processed video',variable=self.tiker4_str)
        tiker4.grid(column=6,row=1)
        
        
        # wont change windows size
        self.update()
        self.geometry(self.geometry())
        
        self.entry.focus_set()
        #self.entry.select_range(0,Tkinter.END)
        
    have_images=0    
    cam_on=0
    pre_tiker1=0
    pre_tiker2=0
    pre_tiker3=0
    def load_images(self):
        self.location=self.entryVariable.get()
        self.load_image(self.location)
        if self.file_names==[]:
            return
        self.image_list=self.file_names
        self.img_idx_str.set(str(self.cur_image_idx+1)+'/'+str(len(self.image_list)))
        self.image=ImageTk.PhotoImage(self.image_resize(self.image_list[0]))
        self.label2.configure(image=self.image)
        self.have_images=1
        
    def OnButtonClick(self):
        self.load_images()
        
    def Button_next(self):
        if self.cam_on:
            self.recog_str.set('right')
            self.recog.configure(background='red')
        if self.have_images==0:
            return 
        self.cur_image_idx+=1
        if self.cur_image_idx==len(self.image_list):
            self.cur_image_idx=0
        # change image
        self.image=ImageTk.PhotoImage(self.image_resize(self.image_list[self.cur_image_idx]))
        self.label2.configure(image=self.image)
        # change idx
        self.img_idx_str.set(str(self.cur_image_idx+1)+'/'+str(len(self.image_list)))
        
    
    def Button_pre(self):
        if self.cam_on:
            self.recog_str.set('left')
            self.recog.configure(background='red')
        if self.have_images==0:
            return 
        self.cur_image_idx-=1
        if self.cur_image_idx<0:
            self.cur_image_idx=len(self.image_list)-1
            
        self.image=ImageTk.PhotoImage(self.image_resize(self.image_list[self.cur_image_idx]))
        self.label2.configure(image=self.image)
        # change idx
        self.img_idx_str.set(str(self.cur_image_idx+1)+'/'+str(len(self.image_list)))
        
        
    def OnPressEnter(self,event):
        self.load_images()

    def Button_open_cam(self):
        if self.cam_on==1:
            return
        width, height = 640, 480
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cam_on=1
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=100)
        self.show_frame()
        #self.webcam('haarcascade_eye.xml',eye=1,hand=1,show=1)
        

    def Button_exit(self):
        self.Button_close_cam()
        app.destroy()

    
    def Button_close_cam(self):
        if self.cam_on==1:
            self.cap.release()
        self.cam_on=0
        self.cam.configure(image=self.cam_img)
        cv2.destroyAllWindows()
        self.state_eye=0
        self.state_hand=0
        self.eye_xy=[]
        self.hand_xy=[]
        self.recog_str.set('None')
        self.recog.configure(background='blue')

    
if __name__ == "__main__":
    app = simpleapp_tk(None)
    app.title('my application')
    app.mainloop()




#webcam_detect('F:/thesis/CSIQ/ref_orig/')
#face1='haarcascade_frontalface_alt.xml'
#eye='haarcascade_eye.xml'
#webcam(eye,eye=1,hand=1)
#load_image('F:/thesis/CSIQ/ref_orig/')