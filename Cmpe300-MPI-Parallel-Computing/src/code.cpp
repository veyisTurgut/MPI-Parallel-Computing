/* Student Name: Adalet Veyis Turgut
Student Number: 2017400210
Compile Status: Compiling
Program Status: Working
Notes:  To compile: mpic++ -o cmpe300_mpi_2017400210 ./cmpe300_mpi_2017400210.cpp
        To run: mpirun  --oversubscribe -np <P> cmpe300_mpi_2017400210 <inputfile>
            If you encounter extra lines before and after the expected results please add the following command while running:
            --mca btl_vader_single_copy_mechanism none
        Input file must be a ".tsv" file, otherwise program can't read the lines.
        I used OpenMPI version 4.0.3
 */

#include <iostream> // std::cout
#include <mpi.h>
#include <stdlib.h>
#include <fstream>//reading file
#include <vector>
#include <limits>//std::numeric_limits_infinity
#include <algorithm> // std::sort
#include <unistd.h> //sleep
#include <math.h>   //fabs

double manhattanDistance(std::vector<double> vec1, std::vector<double> vec2)
{
    //ManhattanDistance([x1; y1; z1]; [x2; y2; z2]) = |x1 - x2| + |y1 - y2| + |z1 - z2|
    //Accordimg to the formula above, the sum of absolute differences of each indices of two vectors is returned.
    double dist = 0;
    for (int i = 0; i < vec1.size() - 1; i++) //-1 because, last value is class value(0 or 1)
    {
        dist += fabs(vec1[i] - vec2[i]);
    }
    return dist;
}

std::vector<int> topT(std::vector<double> features, int t, int rank)
{
    //features array has the feature weigths. We must extract the indices of max t features.
    std::vector<int> topT(t, 0);               //this will hold indices
    std::vector<std::pair<double, int>> pairs; //we must not forget the initial indices of features while sorting them.
    for (int i = 0; i < features.size(); i++)
    //make pairs features with their initial indices
    {
        pairs.push_back(std::make_pair(features[i], i));
    }
    std::sort(pairs.begin(), pairs.end()); //sort ascending with respect to features
    for (int i = 0; i < t; ++i)
    //extract top T indices
    {
        topT[i] = pairs[features.size() - 1 - i].second;
    }
    //sort these indices
    std::sort(topT.begin(), topT.end());
    std::cout << "Slave P" << rank << " :";
    for (int i = 0; i < t; ++i)
    //print
    {
        std::cout << " " << topT[i];
    }
    std::cout << std::endl;
    return topT;
}

std::vector<int> relief(/*n*(a+1) length array*/ double *instancesArray, /*n: #of instances*/ int n, /*n: #of features*/ int a, /*n: #of iterations*/ int m, /*Top t features will be printed*/ int t, int rank)
{
    /*Since I scattered 1d array to slaves and vector is easy to work with, I convert that array to vector of vectors.
    Since there are N lines of input, there are N vector of vectors. 
    Each one of these N vectors holds a vector of double which consists of A features and a class value.
    */
    std::vector<std::vector<double>> instancesVec(n, std::vector<double>(a + 1, 0));
    for (int i = 0; i < n; i++)
    {
        for (int ii = 0; ii <= a; ii++)
        {
            instancesVec[i][ii] = instancesArray[ii + i * (a + 1)];
        }
    }

    std::vector<double> featureWeights(a, 0); //initialize all feature weights W[A]=0.0
    std::vector<double> nearestHit(a, 0);
    std::vector<double> nearestMiss(a, 0);
    std::vector<double> maxFeature(a, std::numeric_limits<double>::infinity() * -1); //fill mith minimum values
    std::vector<double> minFeature(a, std::numeric_limits<double>::infinity());      //fill with max values

    for (int x = 0; x < a; x++)
    // Find max and min of each feature
    {
        for (int i = 0; i < n; i++)
        {
            if (instancesVec[i][x] < minFeature[x])
            {
                minFeature[x] = instancesVec[i][x];
            }
            if (instancesVec[i][x] > maxFeature[x])
            {
                maxFeature[x] = instancesVec[i][x];
            }
        }
    }

    for (int i = 0; i < m; i++) // p.s: n/p >= m. Otherwise index out of bounds error. Modular (%) might be useful.
    //Find the nearest Miss and nearest hit distances
    {
        // sequantally select instance :  instancesVec[i][];
        double nearestMissDistance = std::numeric_limits<double>::infinity();
        double nearestHitDistance = std::numeric_limits<double>::infinity();
        for (int x = 0; x < n; x++)
        {
            if (x == i)
            {
                continue; //skip current instance
            }
            double manhattanDist = manhattanDistance(instancesVec[x], instancesVec[i]);
            if (instancesVec[x][a] == instancesVec[i][a]) //same class, possible hit
            {
                if (manhattanDist < nearestHitDistance)
                { //Update the hit.
                    nearestHitDistance = manhattanDist;
                    nearestHit = instancesVec[x];
                }
            }
            else //opposite class, possible miss
            {
                if (manhattanDist < nearestMissDistance)
                { //Update the hit.
                    nearestMissDistance = manhattanDist;
                    nearestMiss = instancesVec[x];
                }
            }
        }

        for (int A = 0; A < a; A++)
        //calculate average of weigth for each feature
        {
            //This is the diff function, but I did not write a function for it.
            featureWeights[A] -= (fabs(nearestHit[A] - instancesVec[i][A]) / (maxFeature[A] - minFeature[A])) / m;
            featureWeights[A] += (fabs(nearestMiss[A] - instancesVec[i][A]) / (maxFeature[A] - minFeature[A])) / m;
        }
    }
    //we should return the indices of top T features.
    return topT(featureWeights, t, rank);
}

int main(int argc, char *argv[])
{
    int rank;               //id of the current processor
    int size;               //number of all processors
    MPI_Init(&argc, &argv); //initialize Mpi Environment
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int P, N, A, M, T;                 // input parameters
    std::string line;                  //to hold the lines of the tsv file which is another parameter
    std::ifstream MyReadFile(argv[1]); // Read from the text file
    if (rank == 0)
    //Read the first two lines at master
    {
        std::getline(MyReadFile, line);                          //P, first line of the file is P.
        P = std::stoi(line);                                     //convert string to integer.
        std::getline(MyReadFile, line);                          // N \t A \t M \t T. Get the second line of the file.
        N = std::stoi(line.substr(0, line.find_first_of('\t'))); // integer value of the substring from beginning to index of first tab is N.
        line = line.substr(line.find_first_of('\t') + 1);        // A \t M \t T .Remove the first substring.
        A = std::stoi(line.substr(0, line.find_first_of('\t')));

        line = line.substr(line.find_first_of('\t') + 1); // M \t T
        M = std::stoi(line.substr(0, line.find_first_of('\t')));
        line = line.substr(line.find_first_of('\t') + 1); // T
        T = std::stoi(line.substr(0, line.find_first_of('\t')));
    }
    //Broadcast these values to slaves.
    MPI_Bcast(&P, 1, MPI_INT, 0, MPI_COMM_WORLD); // broadcast P
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD); // broadcast N
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD); // broadcast M
    MPI_Bcast(&A, 1, MPI_INT, 0, MPI_COMM_WORLD); // broadcast A
    MPI_Bcast(&T, 1, MPI_INT, 0, MPI_COMM_WORLD); // broadcast T
    MPI_Barrier(MPI_COMM_WORLD);                  // synchronizing processes

    /*
    Initialize the necessary variables within all processors to avoid possible errors of scatter and gather methods.
    Since scatter also sends the data to master processor again, I needed to add extra 0's to the beginning of the arrays.
    */
    double inputData[(N / (P - 1) + N) * (A + 1)];  /* Think it like a 2d array where each consecutive a+1 elements are features and class value.
    First N/(P-1)*(A+1) space is allocated for by master processor, won't be used. Other spaces will be distributed to each processor. 
    The 1 in A+1 stands for class value. There is a class value( 0 or 1 ) after A features.
    */
    for (int i = 0; i < N / (P - 1) * (A + 1); i++) //when we scatter, first n/(p-1)*(a+1) data will go to master again, to aviod errors we can fill them with 0s.
    {
        inputData[i] = 0;
    }
    double instances[(N / (P - 1)) * (A + 1)]; //slave's local array. Think this array as N/(P-1) lines where each line has A features and a class value.

    int topFeatures[T * P]; /*   This array is for gathering. 
    After calculating top features in relief function, slaves must return these features to master.
    This array holds all those features.
    There are P processors including master and each of them will return T features.
    First T space is for master, again. */
    for (int i = 0; i < T; i++)
    {
        topFeatures[i] = 0;
    }
    int topT[T]; //Slaves will hold the top features in this array and send it to master via gather.

    if (rank == 0)
    //Only master will do these lines.
    {
        for (int i = 0; i < N; i++)
        //Since we already handled first two lines above, there are N lines left.
        {
            getline(MyReadFile, line); //Get the next line from file.
            for (int k = 0; k <= A; k++)
            //Each line has A+1 items. A features and a class value, read it, convert it to integer and store in inputData array.
            {
                inputData[/*we gave this area to master so that scatter works fine*/ (N / (P - 1) * (A + 1)) + /* think it like a 2d array*/ i * (A + 1) + k] = std::stod(line.substr(0, line.find_first_of('\t')));
                line = line.substr(line.find_first_of('\t') + 1);
            }
        }
    }

    /*Now inputData array has all the features and class values. We need to distribute it to slaves.
    First parameter is the data to be sent. Second and fifth parameter is the length of the data to be sent. Third and sixth parameters are the type of the data.
    Fourth parameter is the variable of slaves to hold the incoming data. 
    Last two parameters are straight-forward. */
    MPI_Scatter(
        /*void* send_data*/ inputData,
        /*int send_count*/ (A + 1) * N / (P - 1),
        /*MPI_Datatype send_datatype*/ MPI_DOUBLE,
        /*void* recv_data*/ instances,
        /*int recv_count*/ (A + 1) * N / (P - 1),
        /*MPI_Datatype recv_datatype*/ MPI_DOUBLE,
        /*int root*/ 0,
        /*MPI_Comm communicator*/ MPI_COMM_WORLD);

    if (rank != 0)
    //Only slaves will do these lines.
    {
        std::vector<int> reliefReturnedValue = relief(instances, N / (P - 1), A, M, T, rank); //relief returns top T features in vector form.
        std::copy(reliefReturnedValue.begin(), reliefReturnedValue.end(), topT);              //convert vector to array.
    }

    /* This function is very similar to scatter. Now master will collect data starting from itself. Each slave will send their local topT arrays to master's topFeatures array.
    First T spaces of this array is empty, will not be used as we did in inputData.*/
    MPI_Gather(
        /*void* send_data*/ topT,
        /*int send_count*/ T,
        /*MPI_Datatype send_datatype*/ MPI_INT,
        /*void* recv_data*/ topFeatures,
        /*int recv_count*/ T,
        /*MPI_Datatype recv_datatype*/ MPI_INT,
        /*int root*/ 0,
        /*MPI_Comm communicator*/ MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);                  // synchronizing processes

    if (rank == 0)
    { //Only master will do these lines.
        sleep(1);//I tested the program at least a hundred times. At approximately four or five out of these trials, master was not at the end. So, to be 100% sure I added this line.
        std::cout << "Master P0 :";
        std::sort(topFeatures + T, topFeatures + T * P); //Since gather function just appended all the data coming from slaves, we should sort it.
        int extractedFeatures[T * P];                    // This array is unique. Like a set. Duplicates will be removed. Firstly initialize with -1.
        for (int i = 0; i < T * P; i++)
        {
            extractedFeatures[i] = -1;
        }
        int j = 1;
        extractedFeatures[0] = topFeatures[T];
        for (int i = T; i < T * P; i++)
        //for each value in the sorted topFeatures array skipping first T values.
        {
            if (extractedFeatures[j - 1] != topFeatures[i])
            //if the set has that value, skip. Put to the set, otherwise.
            {
                extractedFeatures[j] = topFeatures[i];
                j++;
            }
        }

        int i = 0;
        while (extractedFeatures[i] != -1)
        //print out the result.
        {
            std::cout << " " << extractedFeatures[i];
            i++;
        }
        std::cout << std::endl;
    }

    MPI_Finalize();

    // Close the file
    MyReadFile.close();

    return 0;
}