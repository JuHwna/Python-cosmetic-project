# 딥러닝을 활용한 화장품 성분분석& 제품인식서비스
## 1. 주관
한국데이터산업진흥원
<br />


## 2. 실시
### 1)장소 : 경희대빅데이터연구소
### 2)프로젝트 진행 날짜 : 2019.08.12 ~ 2019.08.28
<br />


## 3. 프로젝트 구성도
### 1) 팀 이름
#### MSG
![팀이름 소개](https://user-images.githubusercontent.com/49123169/65418701-fc5c8480-de37-11e9-8551-dc134204b4be.PNG)

<br />

### 2) 팀장 및 팀원
|팀장 및 팀원|이름|
|-------|-------------------------------------------|
|팀장|신동준|
|팀원|박주환, 노주영, 김수헌, 정예림, 김소은, 김준형|

<br />


### 3) 각 구성원 역할
|역할|이름|
|----------------------------------------|-------------------------------------------|
|DB 구축 및 웹사이트 구축|신동준|
|웹사이트 구축 및 웹사이트 디자인|김준형|
|성분 크롤링 및 인스타크롤링, 모델링|박주환|
|화장품 크롤링 및 전처리, 발표|정예림|
|화장품 크롤링 및 전처리, PPT제작|김소은|
|화장품 크롤링 및 전처리, PPT제작|김수헌|
|화장품 크롤링 및 전처리, PPT제작, 모델학습|노주영|
<br />

### 4) 시스템 아키텍처
![시스템 아키텍처2](https://user-images.githubusercontent.com/49123169/72203117-f9e76e80-34aa-11ea-9a89-6bc2eb66af84.PNG)
* AWS 환경에서 해당 프로젝트를 진행했습니다.

|System|사용|
|----------------------------------------|-------------------------------------------|
|AWS 사용환경| Amazon SageMaker|
|WEB |Flask|
|Deep Learning|Keras|
|DataBase|MySQL|
|Data preprocessing|OpenCV, Pandas, NumPy, PIL|
|Crawling|Selenium, BeautifulSoup|
<br />

### 5) 화장품 서비스
#### * 이름 : 화분
![화분](https://user-images.githubusercontent.com/49123169/65417295-b18d3d80-de34-11e9-839e-7aa1304072ad.png)
<br />

#### * 작동 방법
![시스템 아키텍처](https://user-images.githubusercontent.com/49123169/72203031-ceb04f80-34a9-11ea-8dcf-44019f948cd4.PNG)
* 휴대폰으로 화장품을 사진으로 촬영하여 이미지를 업로드합니다. 
* 업로드된 이미지를 분석하여 정확한 제품을 인식하고 그 제품의 화장품 성분을 찾아 매칭시킵니다.
* 매칭시킨 정보를 사용자에게 바로 알려주는 순서로 진행됩니다.
<br />

## 4. 프로젝트 목표
1. 사회적 목표 : 각 화장품에 대한 성분을 제대로 알려주어 사람들이 자신에게 맞는 화장품을 쓰게 만들기 위해서입니다.
2. 팀적 목표 : 자신이 할 수 있는 프로그램에 대한 더 깊은 공부 및 복습, 해당 교육으로 배운 CNN을 구현하는데 목표를 두었습니다.
<br />

## 5. 프로젝트 진행
### 1) 데이터 수집 
(1) 화장품 성분 수집(성분 등급표)
* 화장품 성분에 대한 등급을 수집하기 위해 EWG라는 미국의 비영리 단체의 자료를 수집했습니다.
![ewg 그림](https://user-images.githubusercontent.com/49123169/72205661-3f1a9900-34c9-11ea-894d-0e913900601d.png)

* 이 사이트에는 1~10등급까지 성분에 대한 등급표가 나오기 때문에 저희 프로젝트에 적합한 자료라고 생각하여 크롤링을 통해 자료를 수집했습니다. 크롤링으로는 **Selenium**과 **Beautifulsoup**을 사용했습니다. 
  * **Selenium**을 사용한 이유는 ewg의 사이트에서 성분을 수집하기 위해선 화장품을 클릭한 후에 성분을 수집해야 합니다. 동적 크롤링에서는 selenium이 최적이기 때문에 크롤링 속도가 느리더라고 이 방법을 사용했습니다.
  * **Beautifulsoup**을 사용한 이유는 화장품 성분 등급표를 가지고 오려고 했기 때문입니다. 성분 등급표가 이미지파일로 되어 있기 때문에 이를 숫자만 가지고 오기 위해서는 이미지 링크를 통해 얻어야 했습니다. selenium의 경우 링크에 대한 정보를 가져오는 것이 힘들었습니다. 그에 비해 beautifulsoup은 링크 가져오는 것이 꽤 쉬웠고 분리하는 과정도 쉬웠습니다. 그래서 등급표만 beautifulsoup을 사용했습니다.
<br />

~~~
#크롤링을 위해 짠 코드
for j in range(0,10000,10):
    driver.implicitly_wait(randint(2,4))
    driver.get("""https://www.ewg.org/skindeep/browse.php?category=after_sun_product&&showmore=products&start={0}""".format(j))
    for i in range(2,12):
        driver.find_element_by_xpath("""//*//*[@id="table-browse"]/tbody/tr[{0}]/td[3]/a""".format(i)).click()
        a_element=driver.find_elements_by_xpath("//td[@class='firstcol'] |a[@href]")
        ic=[]
        for i in range(len(a_element)):
            ic.append(a_element[i].text)
        ic=[x for x in ic if x]
        span_element=driver.find_elements_by_xpath("//td[@width]  |span[@style]")
        sb=[]
        for i in range(len(span_element)):
            sb.append(span_element[i].text)
        sb=[x for x in sb if x]
        div_element=driver.find_elements_by_xpath("//td[@align]  |div[@*]")
        ssb=[]
        for i in range(len(div_element)):
            ssb.append(div_element[i].text)
        ssb=[x for x in ssb if x]
        html=driver.page_source
        soup=BeautifulSoup(html,'html.parser')
        data1_list=soup.findAll('div',{'id':"prod_cntr_score"})
        li_list=[]
        for data1 in data1_list:
            li_list.extend(data1.findAll('img'))
        li_list2=[]
        for x in li_list:
            li_list2.append(str(x))
        li_list3=[]
        for y in range(len(li_list2)):
            li_list3.append(li_list2[y][68:69])
        ic=pd.Series(ic,name='ingredient')
        sb=pd.Series(sb,name='concerns')
        ssb=pd.Series(ssb,name='score')
        li_list=pd.Series(li_list3,name='score_number')
        first=pd.concat([ic,sb,ssb,li_list],axis=1)
        ewg=ewg.append(first)
        driver.back() 
 ~~~
<br />


(2) 화장품 성분 수집(분류할 화장품 성분) 
* 7개의 브랜드(CHANEL, BOBBI BROWN, ESTEE LAUDER, ETUDE HOUSE, innisfree, MISSHA, Dr.Jart), 
  5개의 제품군(스킨,로션, 선크림, 클렌징, 크림)을 사용하여 **33개의 화장품**을 분류할 계획을 세웠습니다.
* 그래서 해당 제품의 성분들을 추출하기 위해서 각 홈페이지에 들어가 크롤링 작업을 실시했습니다.
* 해당 작업은 다른 팀원들이 진행하여 코드를 가지고 있지 않습니다.
![화장품 성분 이름](https://user-images.githubusercontent.com/49123169/72216271-61142a00-3562-11ea-9cdd-d580c9ddbd7a.PNG)
<br />


(3) 화장품 이미지 크롤링(구글 이미지) 
* 33개의 화장품 사진을 학습시킬 이미지를 구하기 위해 첫 번째 방법으로 구글을 사용했습니다.
   구글 검색을 통해 얻을 수 있는 이미지를 추출했습니다.
  (제 담당이 아니라서 코드가 없습니다.)
![구글 이미지](https://user-images.githubusercontent.com/49123169/72216791-fca89900-3568-11ea-8245-c538b7c47db1.PNG)
<br />


(4) 화장품 이미지 크롤링(인스타그램)
* 학습시킬 화장품 사진의 이미지가 매우 부족하여 두 번째 방법으로 인스타그램 크롤링을 진행했습니다.
* https://github.com/huaying/instagram-crawler 해당 깃허브에 있는 방법을 가지고 살짝 튜닝 과정을 거쳐 크롤링을 진행했습니다.
<br />


### 2) 데이터 전처리
(1) 화장품 성분(EWG) 전처리
* EWG 등급표의 경우, 겹치는 성분이 있다는 것을 확인했습니다. 그래서 파이썬을 통해 중복을 없앴습니다.
* 그 후, 영어 성분을 한글로 변역하는 과정을 거쳤습니다. 한글로 번역은 인터넷에서 영어 성분과 한글 성분이 같이 있는 파일을 구해서
  파이썬을 통해 일치성 여부를 판단하여 번역을 진행했습니다.
* 또한 중국어를 할 줄 아는 팀원이 있어서 성분을 중국어로 바꾸는 작업도 거쳤습니다.
<br />


(2) 제품 데이터 DB화
* 제품에 관련된 성분이나 이름 등을 DB에 올려놓기 위해서 일정한 양식으로 통일하는 과정을 거쳤습니다.
* 예시

|No|회사명|제품명|제품군|가격|성분|
|----|-------|------|------|----|----|
|1|바비브라운|스킨 리바이|1|가격|성분|
<br />


(3) 화장품 이미지 전처리
* 수집한 화장품 이미지 데이터를 다 사용하기에는 부적합했습니다. 그 이유가 사람 얼굴, 인형 얼굴 등 화장품에 비해 과도하게 이미지에서
  차지하는 비중이 많은 사진들이 많았습니다. 그래서 이것들을 삭제하거나 일부분만 수집하는 과정을 거쳤습니다.
1. **OpenCV**를 활용
https://github.com/opencv/opencv/tree/master/data/haarcascades
해당 깃허브 주소에서 사람 얼굴을 찾는데 도움을 주는 data가 있습니다. 해당 data를 이용하여 사람 얼굴이 들어간 사진들을 모두 추출했습니다.

~~~~
for i in range(len(drjart)):
    path1 = "./djart/{}".format(drjart[i])
    file_list1 = os.listdir(path1)

    # file_list_jpg = [file for file in file_list1 if file.endswith(".jpg")]
    # print ("file_list: {}".format(file_list_jpg))
    file_list_jpg = [file for file in file_list1 if file.endswith(".jpg") ]
    for i in range(len(file_list_jpg)):
        image = cv2.imread(path1+'/'+'{0}'.format(file_list_jpg[i]))
        image_gs=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces1=len(face_cascade1.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150)))
        faces2=len(face_cascade2.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150))) 
         faces3=len(face_cascade3.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150)))
         faces4=len(face_cascade4.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150)))
         faces5=len(face_cascade5.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150)))
         faces6=len(face_cascade6.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150)))
         faces7=len(face_cascade7.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=5, minSize=(150, 150)))
        if sum([faces1,faces2])>0:
            shutil.move(path1+'/'+'{0}'.format(file_list_jpg[i]),path1+'/face/'+'{0}'.format(file_list_jpg[i])) 
         if faces2>0:
             shutil.move(path1+'/'+'{0}'.format(file_list_jpg[i]),path1+'/face/'+'{0}'.format(file_list_jpg[i]))
~~~~~
<br />

2. 일일이 이미지 전처리
* 이 후에 나머지 이미지는 직접 눈으로 보며 전처리를 진행했습니다.

3. 이미지 사이즈 조정
* 학습시킬 이미지가 모두 같은 크기여야지 학습할 수 있습니다. 그러므로 이를 맞추는 작업이 필요했습니다.
* 사진의 크기는 250x250으로 정해놓았습니다. 250x250으로 바꿀 때 정사이즈로 줄이거나 늘려서 이미지 크기를 맞추었습니다.
  늘리거나 줄일 때 나타나는 빈 공간은 검정색 바탕을 추가하여 공간을 매꾸었습니다.
  
~~~
workDIr = os.path.abspath('./drjart/')
size = 250, 250 
for (path, dirname,files) in os.walk(workDIr):
    os.mkdir(path+'/250x250')
    for f in files: 
        ext = os.path.splitext(f) 
        upper_ext = ext[1] 
        outfile = os.path.join(path+'/250x250', f.split('.')[0]+ '_300x300' + upper_ext)
        try: 
            new_img = Image.new("RGB", (250,250), "black") 
            im = Image.open(os.path.join(path,f)) 
            im.thumbnail(size, Image.ANTIALIAS) 
            load_img = im.load() 
            load_newimg = new_img.load() 
            i_offset = (300 - im.size[0]) / 2 
            j_offset = (300 - im.size[1]) / 2 
            for i in range(0, im.size[0]): 
                for j in range(0, im.size[1]): 
                    load_newimg[i + i_offset,j + j_offset] = load_img[i,j] 
                if(upper_ext == '.JPEG' or upper_ext == '.JPG'): 
                    new_img.save(outfile, "JPEG") 
                elif(upper_ext == '.png'): 
                    new_img.save(outfile, "png") 
        except IOError: 
            print( "cannot create thumbnail for '%s'" % os.path.join(path, f))
~~~

## 6. 모델 구축
* 전처리한 결과, 얻은 화장품 이미지 개수가 각각 70개 정도였습니다. 이 정도의 양으로는 학습시키기 부족하다고 판단하여 
  **Image Augmentation(이미지 변조)** 를 통해 학습 데이터를 늘렸습니다.
  
### 1) 초기 모델(자체 제작 모델)  
#### (1) X axis flip & Rotation 
* 먼저 변조 방식으로 **좌우 반전과 회전(90,180,270)** 을 해서 총 이미지 개수를 4배로 늘렸습니다.
  
  *(나중에 깨달은 것은 회전의 경우, 회전해도 외형이 차이가 나지 않는 원이나 정사각형만 회전하는 것이 맞다는 사실을 알게 되었습니다.*
  *컴퓨터의 경우, 외형이 다른 제품에서 1도만 회전해도 다른 물체로 인식한다는 사실......)*
* 그 후, **Keras 기반으로 CNN 모형**을 돌렸습니다.

~~~
# 초창기 모델
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), padding='same',activation='relu', input_shape=(250,250,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, kernel_size=(3,3), padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(33, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=30, epochs=30, verbose=1, validation_split=0.2)
his_list.append(history)
score = model.evaluate(x_test, y_test, verbose=0)
loss_list.append(score[0])
acc_list.append(score[1])
~~~

* 이 후, 여러 변수들을 조절하거나 층을 더 쌓으며 예측률을 올리기 위해 노력했습니다.

|Model|Average Accuracy|
|-------------|----------------|
|3 Conv 3 Fc|0.612|
|5 Conv 3 Fc|0.762|
|7 Conv 5 Fc|0.826|

![첫번째 flip rotation](https://user-images.githubusercontent.com/49123169/72247814-13b6bc00-3639-11ea-8ee1-b7a90f944a81.PNG)

* 최대로 올렸을 때 **82%** 까지 올렸습니다. 하지만 그 이상 올라갈 기미가 보이지 않아 다른 방법이 필요해보였습니다. 
  그래서 다른 방법을 찾기 위해 구글링 작업을 시작했습니다.
~~~
# 80퍼 근처 코드
model = Sequential()
model.add(Conv2D(32,kernel_size=(2,2), padding='same',activation='relu', input_shape=(250,250,3),kernel_initializer='he_normal')))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(2,2), padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(128, kernel_size=(2,2), padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256, kernel_size=(2,2), padding='same',activation='relu',kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(256, kernel_size=(2,2), padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(512, kernel_size=(2,2), padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(1024, kernel_size=(2,2), padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(2048, kernel_size=(2,2), padding='same', activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(1024,activation='relu', kernel_initializer='he_normal'))
model.add(Dense(512,activation='relu', kernel_initializer='he_normal'))
model.add(Dense(256,activation='relu', kernel_initializer='he_normal'))
model.add(Dense(128,activation='relu', kernel_initializer='he_normal'))
model.add(Dense(33, activation='relu', kernel_initializer='he_normal'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=64, epochs=100, verbose=1, validation_split=0.2)
his_list.append(history)
score = model.evaluate(x_test, y_test, verbose=0)
loss_list.append(score[0])
acc_list.append(score[1])
~~~

#### (2) Crop
* 다른 방법을 찾던 도중 **ImageNet Classification with Deep Convolutional Neural Networks** 이라는 논문에서 crop이라는 방법을 볼 수 있었습니다.
>The first form of data augmentation consists of generating image translations and horizontal reflections. We do this by extracting random 224 × 224 patches (and their horizontal reflections) from the 256×256 images and training our network on these extracted patches4
. This increases the size of our training set by a factor of 2048, though the resulting training examples are, of course, highly interdependent. Without this scheme, our network suffers from substantial overfitting, which would have forced us to use much smaller networks. At test time, the network makes a prediction by extracting five 224 × 224 patches (the four corner patches and the center patch) as well as their horizontal reflections (hence ten patches in all), and averaging the predictions made by the network’s softmax
layer on the ten patches.

* Crop은 Image Augmentation 방법 중 하나로 주된 대상을 놔두고 주변을 일정 부분 자르는 방법입니다. 이 방법을 통해 부족한 데이터 셋을 4배로 늘릴 수 있습니다. 해당 이미지 변조 방식이 저희 상황에 딱 맞아 떨어져 바로 이미지 전처리 작업으로 돌아가 crop을 진행했습니다.
~~~
out_220 = imgs[:,15:235,15:235,:]
# print(out.shape)
img1 = imgs[:,:220,:220,:]
img2 = imgs[:,:220,30:250,:]
img3 = imgs[:,30:250,:220,:]
img4 = imgs[:,30:250,30:250,:]
list = [img1,img2,img3,img4]
for i in list:
    print(i.shape)
    out_220 = np.concatenate((out_220,i),axis=0)
~~~

* 원래는 처음에는 crop과 좌우 반전 사진을 다 컴퓨터에 저장하려고 했으나 프로젝트 마감일이 별로 남지 않아서 행렬을 통해 이미지를 부풀렸습니다. 그런데 crop까지는 잘 진행했으나 4배 부풀린 상태에서 flip까지 하려고 하니 aws 이용 환경에 한계를 느끼는 걸 발견했고 colab도 마찬가지라서 4배 부풀린 것만 사용했습니다.
* 또한 기존에 250x250에서 220x220으로 이미지 사이즈가 작아져서 약간의 코드를 수정한 후 모델을 돌렸습니다.
* 이를 초기모델에 사용한 결과, 예측률 80퍼가 나오는 모습을 확인할 수 있었고 가능성이 있다고 보아 여러 번 변수를 조절하고 층을 쌓았더니 95%까지 올릴 수 있었습니다.

|Model|Average Accuracy|
|-------------|----------------|
|3 Conv 3 Fc|0.80|
|5 Conv 5 Fc|0.906|
|6 Conv 5 Fc|0.955|

![image](https://user-images.githubusercontent.com/49123169/72247869-2df09a00-3639-11ea-9576-57401c518410.png)

### 2) VGG16 + Transfer Learning Tuning 
* 만족스러운 예측률을 얻고 그만 두는 과정 속에서 유명한 모델과 어떤 차이가 나는지를 보고 싶었습니다.
* 유명한 모델 중 VGG16 모델을 골라 학습을 진행했습니다.
* 학습된 가중치가 없을 때는 5%도 나오지 않아 모델 잘 구축했다는 생각을 가졌습니다.
* 하지만 ImageNet의 가중치를 가지고 와서 층을 한 두개 더 쌓은 후, 학습을 진행했는데 97%라는 엄청난 예측률을 얻었습니다.

|Model|Average Accuracy|
|-------------|----------------|
|VGG16+Tuning|0.979|

![crop flip vgg16 그래프](https://user-images.githubusercontent.com/49123169/72260421-2b506d80-3656-11ea-8f5d-1fd34859c3a6.PNG)




## 7. 결과
 - 그래프도 과적합이 아니고 실제 다른 이미지로 테스트 한 결과, 10번 중 8번이나 맞추는 놀라운 성능을 보여줬습니다. 이를 통해 해당 모델과 가중치를 화분에 넣을 수 있었고 그로 인해 만족스러운 사이트 및 앱이 만들어졌습니다.
 - 2019 빅데이터 청년인재 프로젝트 발표회에서 우수상을 수상했습니다.
 
## 8. 느낀 점
 - 처음으로 대형 프로젝트를 해서 회사에서 큰 프로젝트 진행하면 이런 식이지 않을까라는 간접 체험을 할 수 있었습니다.
 - 처음으로 CNN 모델을 구축했기 때문에 쉽지 않았지만 많은 것을 배운 의미 있는 경험이었습니다.
 - tensorflow로 짜고 싶었지만 시간 상 keras로 짜서 아쉬웠습니다. 하지만 keras로 모델을 구축하고 학습시킬 때 어떻게 코딩을 해야하는지를
   제대로 배운 뜻 깊은 경험이었습니다.
 - 처음으로 데이터 분석 프로젝트를 하면서 상을 받았습니다. 정말 기분 좋았습니다.
