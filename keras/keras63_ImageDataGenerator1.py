# 폴더를 가져와서 데이터화 해줌
# 폴더에 들어가는 이미지 크기가 일정해야 함

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 이미지에 대한 생성 옵션 정하기
train_datagen = ImageDataGenerator(
    rescale=1./255.,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
    )

test_datagen = ImageDataGenerator(rescale=1./255.)

# flow 또는 flow_from_directory
# 실제 데이터가 있는 곳을 알려주고, 이미지를 불러오는 작업

xy_train = train_datagen.flow_from_directory(
    './data/data1/train',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary'
    # , save_to_dir='./data/data1_2/train'
    )

xy_test = test_datagen.flow_from_directory(
    './data/data1/test',
    target_size=(150,150),
    batch_size=5,
    class_mode='binary'
    )

model.fit_generator(
    xy_train,
    steps_per_epoch = 100,
    epochs = 20,
    validation_data = xy_test,
    validation_steps = 4
)




