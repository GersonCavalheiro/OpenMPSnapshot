!
! Copyright (c) 2013, Intel Corporation
! Copyright (c) 2021, NVIDIA
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions
! are met:
!
! * Redistributions of source code must retain the above copyright
!       notice, this list of conditions and the following disclaimer.
! * Redistributions in binary form must reproduce the above
!       copyright notice, this list of conditions and the following
!       disclaimer in the documentation and/or other materials provided
!       with the distribution.
! * Neither the name of Intel Corporation nor the names of its
!       contributors may be used to endorse or promote products
!       derived from this software without specific prior written
!       permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
! "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, ACLUDAG, BUT NOT
! LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
! FOR A PARTICULAR PURPOSE ARE DISCLAIMED. A NO EVENT SHALL THE
! COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, ADIRECT,
! ACIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (ACLUDAG,
! BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
! LOSS OF USE, DATA, OR PROFITS; OR BUSAESS ATERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER A CONTRACT, STRICT
! LIABILITY, OR TORT (ACLUDAG NEGLIGENCE OR OTHERWISE) ARISAG A
! ANY WAY B OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
! POSSIBILITY OF SUCH DAMAGE.
!
!
! *******************************************************************
!
! NAME:    Stencil
!
! PURPOSE: This program tests the efficiency with which a space-invariant,
!          linear, symmetric filter (stencil) can be applied to a square
!          grid or image.
!
! USAGE:   The program takes as input the linear
!          dimension of the grid, and the number of iterations on the grid
!
!                <progname> <iterations> <grid size>
!
!          The output consists of diagnostics to make sure the
!          algorithm worked, and of timing statistics.
!
! FUNCTIONS CALLED:
!
!          Other than standard C functions, the following functions are used in
!          this program:
!          wtime()
!
! HISTORY: - Written by Rob Van der Wijngaart, February 2009.
!          - RvdW: Removed unrolling pragmas for clarity;
!            added constant to array "in" at end of each iteration to force
!            refreshing of neighbor data in parallel versions; August 2013
!          - Converted to Fortran by Jeff Hammond, January-February 2016.
!
! *******************************************************************
subroutine apply_stencil(is_star,tiling,tile_size,r,n,W,A,B)
use, intrinsic :: iso_fortran_env
implicit none
logical, intent(in) :: is_star, tiling
integer(kind=INT32), intent(in) :: tile_size, r, n
real(kind=REAL64), intent(in) :: W(-r:r,-r:r)
real(kind=REAL64), intent(in) :: A(n,n)
real(kind=REAL64), intent(inout) :: B(n,n)
integer(kind=INT32) :: i, j, ii, jj, it, jt
if (is_star) then
if (.not.tiling) then
do j=r,n-r-1
do i=r,n-r-1
! do not use Intel Fortran unroll directive here (slows down)
do jj=-r,r
B(i+1,j+1) = B(i+1,j+1) + W(0,jj) * A(i+1,j+jj+1)
enddo
do ii=-r,-1
B(i+1,j+1) = B(i+1,j+1) + W(ii,0) * A(i+ii+1,j+1)
enddo
do ii=1,r
B(i+1,j+1) = B(i+1,j+1) + W(ii,0) * A(i+ii+1,j+1)
enddo
enddo
enddo
else ! tiling
do jt=r,n-r-1,tile_size
do it=r,n-r-1,tile_size
do j=jt,min(n-r-1,jt+tile_size-1)
do i=it,min(n-r-1,it+tile_size-1)
do jj=-r,r
B(i+1,j+1) = B(i+1,j+1) + W(0,jj) * A(i+1,j+jj+1)
enddo
do ii=-r,-1
B(i+1,j+1) = B(i+1,j+1) + W(ii,0) * A(i+ii+1,j+1)
enddo
do ii=1,r
B(i+1,j+1) = B(i+1,j+1) + W(ii,0) * A(i+ii+1,j+1)
enddo
enddo
enddo
enddo
enddo
endif ! tiling
else ! grid
if (.not.tiling) then
do j=r,n-r-1
do i=r,n-r-1
do jj=-r,r
do ii=-r,r
B(i+1,j+1) = B(i+1,j+1) + W(ii,jj) * A(i+ii+1,j+jj+1)
enddo
enddo
enddo
enddo
else ! tiling
do jt=r,n-r-1,tile_size
do it=r,n-r-1,tile_size
do j=jt,min(n-r-1,jt+tile_size-1)
do i=it,min(n-r-1,it+tile_size-1)
do jj=-r,r
do ii=-r,r
B(i+1,j+1) = B(i+1,j+1) + W(ii,jj) * A(i+ii+1,j+jj+1)
enddo
enddo
enddo
enddo
enddo
enddo
endif ! tiling
endif ! star
end subroutine apply_stencil
program main
use, intrinsic :: iso_fortran_env
use prk
implicit none
integer :: err
! problem definition
integer(kind=INT32) :: iterations                     ! number of times to run the pipeline algorithm
integer(kind=INT32) ::  n                             ! linear grid dimension
integer(kind=INT32) ::  stencil_size                  ! number of points in stencil
integer(kind=INT32) ::  tile_size                     ! loop nest block factor
logical :: tiling                                     ! boolean indication loop nest blocking
logical :: is_star                                    ! true = star, false = grid
integer(kind=INT32), parameter :: r=RADIUS            ! radius of stencil
real(kind=REAL64) :: W(-r:r,-r:r)                     ! weights of points in the stencil
real(kind=REAL64), allocatable :: A(:,:), B(:,:)      ! grid values
real(kind=REAL64), parameter :: cx=1.d0, cy=1.d0
! runtime variables
integer(kind=INT32) :: i, j, k
integer(kind=INT64) :: flops                          ! floating point ops per iteration
real(kind=REAL64) :: norm, reference_norm             ! L1 norm of solution
integer(kind=INT64) :: active_points                  ! interior of grid with respect to stencil
real(kind=REAL64) :: t0, t1, stencil_time, avgtime    ! timing parameters
real(kind=REAL64), parameter ::  epsilon=1.d-8        ! error tolerance
! ********************************************************************
! read and test input parameters
! ********************************************************************
write(*,'(a25)') 'Parallel Research Kernels'
write(*,'(a43)') 'Fortran Serial Stencil execution on 2D grid'
call prk_get_arguments('stencil',iterations=iterations,order=n,tile_size=tile_size)
! TODO: parse runtime input for star/grid
#ifdef STAR
is_star = .true.
#else
is_star = .false.
#endif
tiling = (tile_size.ne.n)
write(*,'(a22,i8)') 'Number of iterations = ', iterations
write(*,'(a22,i8)') 'Grid size            = ', n
write(*,'(a22,i8)') 'Radius of stencil    = ', r
if (is_star) then
write(*,'(a22,a8)')  'Type of stencil      = ', 'star'
stencil_size = 4*r+1
else
write(*,'(a22,a8)')  'Type of stencil      = ','grid'
stencil_size = (2*r+1)**2
endif
if (tiling) then
write(*,'(a22,i8)') 'Tile size            = ', tile_size
else
write(*,'(a10)') 'Tiling off'
endif
! ********************************************************************
! ** Allocate space for the input and perform the computation
! ********************************************************************
allocate( A(n,n), B(n,n), stat=err)
if (err .ne. 0) then
write(*,'(a,i3)') 'allocation returned ',err
stop 1
endif
call initialize_w(is_star,r,W)
do j=1,n
do i=1,n
A(i,j) = cx*i+cy*j
B(i,j) = 0.d0
enddo
enddo
t0 = 0
do k=0,iterations
if (k.eq.1) t0 = prk_get_wtime()
! Apply the stencil operator
call apply_stencil(is_star,tiling,tile_size,r,n,W,A,B)
! add constant to solution to force refresh of neighbor data, if any
do j=1,n
do i=1,n
A(i,j) = A(i,j) + 1.d0
enddo
enddo
enddo ! iterations
t1 = prk_get_wtime()
stencil_time = t1 - t0
norm = 0.0d0
do j=r,n-r
do i=r,n-r
norm = norm + abs(B(i,j))
enddo
enddo
active_points = int(n-2*r,INT64)**2
norm = norm / real(active_points,REAL64)
!******************************************************************************
!* Analyze and output results.
!******************************************************************************
deallocate( A,B )
! verify correctness
reference_norm = real(iterations+1,REAL64) * (cx + cy);
if (abs(norm-reference_norm) .gt. epsilon) then
write(*,'(a,f13.6,a,f13.6)') 'ERROR: L1 norm = ', norm, &
' Reference L1 norm = ', reference_norm
else
write(*,'(a)') 'Solution validates'
#ifdef VERBOSE
write(*,'(a,f13.6,a,f13.6)') 'VERBOSE: L1 norm = ', norm, &
' Reference L1 norm = ', reference_norm
#endif
endif
flops = int(2*stencil_size+1,INT64) * active_points
avgtime = stencil_time/iterations
write(*,'(a,f13.6,a,f13.6)') 'Rate (MFlops/s): ',1.0d-6*flops/avgtime, &
' Avg time (s): ',avgtime
end program main
