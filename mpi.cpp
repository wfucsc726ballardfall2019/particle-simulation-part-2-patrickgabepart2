#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include <vector>
#include "omp.h"
using namespace std;

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg; 
 
    //
    //  process command line parameters
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;


    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( sizeof(particle_t), MPI_BYTE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );
    
    //
    //  set up the data partitioning across processors
    //
    int particle_per_proc = (n + n_proc - 1) / n_proc;
    int *partition_offsets = (int*) malloc( (n_proc+1) * sizeof(int) );
    for( int i = 0; i < n_proc+1; i++ ){
        partition_offsets[i] = min( i * particle_per_proc, n );
        // cout << partition_offsets[i] << " ";
    }
    // cout << endl;
    
    int *partition_sizes = (int*) malloc( n_proc * sizeof(int) );
    for( int i = 0; i < n_proc; i++ ){
        partition_sizes[i] = partition_offsets[i+1] - partition_offsets[i];
        // cout << partition_sizes[i] << " ";
    }
    // cout << endl;
    
    //
    //  allocate storage for local partition
    //
    int nlocal = partition_sizes[rank];
    particle_t *local = (particle_t*) malloc( nlocal * sizeof(particle_t) );
    
    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    set_size( n );
    if( rank == 0 )
        init_particles( n, particles );
    
    MPI_Scatterv( particles, partition_sizes, partition_offsets, PARTICLE, local, nlocal, PARTICLE, 0, MPI_COMM_WORLD );

    double cutoff = 0.02;
    double space_dim = sqrt(n * 0.0005);
    // cout << "Actual space is " << space_dim << " by " << space_dim << endl;
    double cell_edge = space_dim / floor(space_dim / cutoff);
    // cell_edge = space_dim / 4;
    // cout << "Cells are size " << cell_edge << " by " << cell_edge << endl;
    int num_cells = pow((space_dim / cell_edge), 2);
    // cout << num_cells << endl;
    int cells_in_row = (int)sqrt(num_cells);
    vector<vector<int> > cell_vector(n_proc);
    // cout << cells_in_row << endl;
    vector<particle_t> particles_to_send;
    vector<particle_t> particles_to_receive;
    vector<particle_t> local_vec;

    int count = 0;
    for(int i = 0; i < nlocal; i++){
      // cout << rank << " ";
      // cout << local[i].x << " " << local[i].y << "   midpoint: " << space_dim / 2 << endl;
      if(rank == 0 && local[i].x >= space_dim / 2){
        if(local[i].x != 0 && local[i].y != 0){
          // cout << "\tSEND" << endl;
          particles_to_send.push_back(local[i]);
          // cout << rank << " " << local[i].x << " " << local[i].y << endl;
          // cout << rank << " " << local[i].index << " " << local[i].x << " " << local[i].y << endl;
          count++;
        }
      }
      else if(rank == 1 && local[i].x < space_dim / 2){
        if(local[i].x != 0 && local[i].y != 0){
          // cout << "\tSEND" << endl;
          particles_to_send.push_back(local[i]);
          // cout << rank << " " << local[i].x << " " << local[i].y << endl;
          // cout << rank << " " << local[i].index << " " << local[i].x << " " << local[i].y << endl;
          count++;
        }
      }
      else{
          local_vec.push_back(local[i]);
      }

      
      
      // if(particles[i].x * n >= partition_offsets[rank+1] - (cutoff * n)){
      // if(rank == 0 && (local[i].x * n >= (partition_offsets[rank+1] - (cutoff * n)))){
      //   cout << rank << " " << local[i].x * n << " " << local[i].y * 2000 << endl;
      // }
      // else if(rank == 1 && (local[i].x * n <= (partition_offsets[rank] + (cutoff * n)))){
      //   cout << rank << " " << local[i].x * n << " " << local[i].y * 2000 << endl;
      // }
    }

    // cout << "Thread " <<  rank << " first contains: ";
    // for(int i = 0; i < nlocal; i++){
    //   cout << local[i].index << " ";
    // }
    // cout << endl;

    // cout << "Thread " <<  rank << " particles to send: ";
    // for(int i = 0; i < particles_to_send.size(); i++){
    //   cout << particles_to_send[i].index << " ";
    // }
    // cout << endl;

    int num_receive;
    if(rank == 0){

      MPI_Send(&count, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
      MPI_Send(&particles_to_send[0], particles_to_send.size(), PARTICLE, 1, 0, MPI_COMM_WORLD);
      MPI_Recv(&num_receive, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      particles_to_receive.resize(num_receive);
      MPI_Recv(&particles_to_receive[0], num_receive, PARTICLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // cout << "Thread " <<  rank << " particles received: ";
      for(int i = 0; i < particles_to_receive.size(); i++){
        // cout << particles_to_receive[i].index << " ";
        local_vec.push_back(particles_to_receive[i]);
      }
      // cout << endl;

      // cout << "Thread 0 now contains: ";
      // for(int i = 0; i < local_vec.size(); i++){
      //   cout << local_vec[i].index << " ";
      // }
      // cout << endl;

    }
    else{ //Thread 1

      MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
      MPI_Send(&particles_to_send[0], particles_to_send.size(), PARTICLE, 0, 0, MPI_COMM_WORLD);
      MPI_Recv(&num_receive, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      particles_to_receive.resize(num_receive);
      MPI_Recv(&particles_to_receive[0], num_receive, PARTICLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // cout << "Thread " <<  rank << " particles received: ";
      for(int i = 0; i < particles_to_receive.size(); i++){
        // cout << particles_to_receive[i].index << " ";
        local_vec.push_back(particles_to_receive[i]);
      }
      // cout << endl;

      // cout << "Thread 1 now contains: ";
      for(int i = 0; i < local_vec.size(); i++){
        // cout << local_vec[i].index << " ";
      }
      // cout << endl;

    }

    vector<particle_t> temp;
    vector<particle_t> boundary;
    vector<particle_t> b_temp;
    
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
        // cout << endl;

        // cout << "Thread " <<  rank << " first contains: ";
        // for(int i = 0; i < local_vec.size(); i++){
        //   cout << local_vec[i].index << " ";
        // }
        // cout << endl;

        
        particles_to_receive.clear();
        particles_to_send.clear();
        temp.clear();
        boundary.clear();
        b_temp.clear();

        int count = 0;
        int bcount = 0;
        for(int i = 0; i < local_vec.size(); i++){
          // cout << rank << " ";
          // cout << local[i].x << " " << local[i].y << "   midpoint: " << space_dim / 2 << endl;
          if(rank == 0 && local_vec[i].x >= space_dim / 2){
            if(local_vec[i].x != 0 && local_vec[i].y != 0){
              // cout << "\tSEND" << endl;
              particles_to_send.push_back(local_vec[i]);
              // cout << rank << " " << local[i].x << " " << local[i].y << endl;
              // cout << rank << " " << local[i].index << " " << local[i].x << " " << local[i].y << endl;
              count++;
            }
          }
          else if(rank == 1 && local_vec[i].x < space_dim / 2){
            if(local_vec[i].x != 0 && local_vec[i].y != 0){
              // cout << "\tSEND" << endl;
              particles_to_send.push_back(local_vec[i]);
              // cout << rank << " " << local[i].x << " " << local[i].y << endl;
              // cout << rank << " " << local[i].index << " " << local[i].x << " " << local[i].y << endl;
              count++;
            }
          }
          else{
              if(local_vec[i].x >= (space_dim / 2) - cutoff && local_vec[i].x <= (space_dim / 2) + cutoff){
                // cout << "WHOA BUDDY" << endl;
                bcount++;
                boundary.push_back(local_vec[i]);
              }
              temp.push_back(local_vec[i]);
          }
        }
        local_vec = temp;

        // cout << "Thread " <<  rank << " particles to send: ";
        // for(int i = 0; i < particles_to_send.size(); i++){
        //   cout << particles_to_send[i].index << " ";
        // }
        // cout << "( boundary: ";
        // for(int i = 0; i < boundary.size(); i++){
        // cout << boundary[i].index << " ";
        // }
        // cout << ")" << endl;

        

        int num_receive;
        int num_b;
        if(rank == 0){

          MPI_Send(&count, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
          MPI_Send(&particles_to_send[0], particles_to_send.size(), PARTICLE, 1, 0, MPI_COMM_WORLD);

          MPI_Send(&bcount, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
          MPI_Send(&boundary[0], boundary.size(), PARTICLE, 1, 0, MPI_COMM_WORLD);

          MPI_Recv(&num_receive, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          particles_to_receive.resize(num_receive);
          MPI_Recv(&particles_to_receive[0], num_receive, PARTICLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          MPI_Recv(&num_b, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          b_temp.resize(num_b);
          MPI_Recv(&b_temp[0], num_b, PARTICLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        }
        else{ //Thread 1

          MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
          MPI_Send(&particles_to_send[0], particles_to_send.size(), PARTICLE, 0, 0, MPI_COMM_WORLD);

          MPI_Send(&bcount, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
          MPI_Send(&boundary[0], boundary.size(), PARTICLE, 0, 0, MPI_COMM_WORLD);

          MPI_Recv(&num_receive, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          particles_to_receive.resize(num_receive);
          MPI_Recv(&particles_to_receive[0], num_receive, PARTICLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

          MPI_Recv(&num_b, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          b_temp.resize(num_b);
          MPI_Recv(&b_temp[0], num_b, PARTICLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // cout << "Thread " <<  rank << " particles received: ";
        for(int i = 0; i < particles_to_receive.size(); i++){
          // cout << particles_to_receive[i].index << " ";
          local_vec.push_back(particles_to_receive[i]);
        }
        // cout << "( boundary: ";
        // for(int i = 0; i < b_temp.size(); i++){
        // cout << b_temp[i].index << " ";
        // }
        // cout << ")" << endl;

        // cout << "Thread " << rank << " now contains: ";
        // for(int i = 0; i < local_vec.size(); i++){
        //   cout << local_vec[i].index << " ";
        // }
        // for(int i = 0; i < b_temp.size(); i++){
        //   cout << b_temp[i].index << " ";
        // }
        // cout << endl;

        MPI_Barrier(MPI_COMM_WORLD);

        navg = 0;
        dmin = 1.0;
        davg = 0.0;
        // 
        //  collect all global data locally (not good idea to do)
        //
        // MPI_Allgatherv( local, nlocal, PARTICLE, particles, partition_sizes, partition_offsets, PARTICLE, MPI_COMM_WORLD );
        
        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        if( find_option( argc, argv, "-no" ) == -1 )
          if( fsave && (step%SAVEFREQ) == 0 )
            save( fsave, n, particles );
        
        //
        //  compute all forces
        //
        for( int i = 0; i < local_vec.size(); i++ )
        {
            local_vec[i].ax = local_vec[i].ay = 0;
            for (int j = 0; j < local_vec.size(); j++ )
            {
              // if(i < j){
                // cout << "\tApplying " << local_vec[i].index << " to " << local_vec[j].index << endl;
                apply_force( local_vec[i], local_vec[j], &dmin, &davg, &navg );
              // }
            }
            for (int j = 0; j < b_temp.size(); j++){
              // cout << "\tApplying " << local_vec[i].index << " to " << b_temp[j].index << endl;
              apply_force( local_vec[i], b_temp[j], &dmin, &davg, &navg );
            }
        }
     
        MPI_Barrier(MPI_COMM_WORLD);

        if( find_option( argc, argv, "-no" ) == -1 )
        {
          
          MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);

 
          if (rank == 0){
            //
            // Computing statistical data
            //
            if (rnavg) {
              absavg +=  rdavg/rnavg;
              nabsavg++;
            }
            if (rdmin < absmin) absmin = rdmin;
          }
        }

        //
        //  move particles
        //
        for( int i = 0; i < local_vec.size(); i++ )
            move( local_vec[i] );

        MPI_Barrier(MPI_COMM_WORLD);
    }
    simulation_time = read_timer( ) - simulation_time;
  
    if (rank == 0) {  
      printf( "n = %d, simulation time = %g seconds", n, simulation_time);

      if( find_option( argc, argv, "-no" ) == -1 )
      {
        if (nabsavg) absavg /= nabsavg;
      // 
      //  -The minimum distance absmin between 2 particles during the run of the simulation
      //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
      //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
      //
      //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
      //
      printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
      if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
      if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
      }
      printf("\n");     
        
      //  
      // Printing summary data
      //  
      if( fsum)
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }
  
    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    free( partition_offsets );
    free( partition_sizes );
    free( local );
    free( particles );
    if( fsave )
        fclose( fsave );
    
    MPI_Finalize( );
    
    return 0;
}
