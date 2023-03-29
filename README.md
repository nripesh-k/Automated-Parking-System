# Automated-Parking-System

The Automated Parking System is an application that automates the process of parking management for vehicles. It uses sensors, cameras, and other technologies to detect the vehicles numberplate and available parking spaces, eliminating the need for human intervention in the parking management process.

## Features

- Number Plate Recognition through CNN
- Real-time Parking Lot monitoring for occupancy

## Methods

- Numberplate Localization
The numberplate localization of the image is reliant on image filtering through openCV. Filters like Blurring, thresholding, eroding, etc have been used to then apply the connected components algorithm through openCV. The resulting components are further filtered to find the cluster of numberplate characters.

Example Image:

<img src="https://github.com/nripesh-k/Automated-Parking-System/blob/main/readmeImages/1.jpg?raw=true" width = '500' height = '400'>

Otus Thresholding:

<img src="https://github.com/nripesh-k/Automated-Parking-System/blob/main/readmeImages/otus.png?raw=true" width = '500' height = '400'>

Filtered Connected Components:

<img src="https://github.com/nripesh-k/Automated-Parking-System/blob/main/readmeImages/otusconnection.png?raw=true" width = '500' height = '400'>

NumberPlate:

![NumberPlate](https://github.com/nripesh-k/Automated-Parking-System/blob/main/readmeImages/1_0.jpg?raw=true)

Character components in Numberplate:

![Characters](https://github.com/nripesh-k/Automated-Parking-System/blob/main/readmeImages/letterComponents.png?raw=true)

Normalized characters:

![1](https://github.com/nripesh-k/Automated-Parking-System/blob/main/readmeImages/1_1.jpg?raw=true)
![2](https://github.com/nripesh-k/Automated-Parking-System/blob/main/readmeImages/1_2.jpg?raw=true)
![3](https://github.com/nripesh-k/Automated-Parking-System/blob/main/readmeImages/1_3.jpg?raw=true)
![4](https://github.com/nripesh-k/Automated-Parking-System/blob/main/readmeImages/1_4.jpg?raw=true)
![5](https://github.com/nripesh-k/Automated-Parking-System/blob/main/readmeImages/1_5.jpg?raw=true)
![6](https://github.com/nripesh-k/Automated-Parking-System/blob/main/readmeImages/1_6.jpg?raw=true)
![7](https://github.com/nripesh-k/Automated-Parking-System/blob/main/readmeImages/1_7.jpg?raw=true)
![8](https://github.com/nripesh-k/Automated-Parking-System/blob/main/readmeImages/1_8.jpg?raw=true)



## Future Enhancements
- Extend to all Devnagri characters in Nepali Lisence Plate
- Easy Connection to a managed database
- Latency improvement

The image dataset was retrieved from: https://github.com/Prasanna1991/LPR

Citation for the dataset:

@inproceedings{pant2015automatic,
  title={Automatic Nepali Number Plate Recognition with Support Vector Machines},
  author={Pant, Ashok Kumar and Gyawali, Prashnna Kumar and Acharya, Shailesh},
  booktitle={Proceedings of the 9th International Conference on Software, Knowledge, Information Management and Applications (SKIMA)},
  pages={92--99},
  year={2015}
}
