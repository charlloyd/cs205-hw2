!
!     Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
!
! NVIDIA CORPORATION and its licensors retain all intellectual property
! and proprietary rights in and to this software, related documentation
! and any modifications thereto.  Any use, reproduction, disclosure or
! distribution of this software and related documentation without an express
! license agreement from NVIDIA CORPORATION is strictly prohibited.
! 

      program verify

      integer i, nerrors
      real*8 r1, r2, r1_expd, r2_expd

      open (unit=8,file='energy32_32_32_150.dat.expd')
      open (unit=9,file='energy32_32_32_150.dat')

      nerrors = 0
      do i = 1, 150
        read(8,*) r1_expd, r2_expd
        read(9,*) r1, r2
        if (abs(r1_expd) .lt. 1.0d-6) then

          if ( abs(r1_expd - r1) .gt. 1.0d-10  ) then
            nerrors = nerrors + 1
          endif    

        else

        if ((abs(r1_expd-r1)/min(abs(r1_expd),abs(r1))).ge.1.0d-6) then
            nerrors = nerrors + 1
          endif    

        endif

        if (abs(r2_expd) .lt. 1.0d-6) then

          if ( abs(r2_expd - r2) .gt. 1.0d-10  ) then
            nerrors = nerrors + 1
          endif    

        else

        if ((abs(r2_expd-r2)/min(abs(r2_expd),abs(r2))).ge.1.0d-6) then
            nerrors = nerrors + 1
          endif    

        endif

      enddo

      if (nerrors .ne. 0) then
        print *, "Test FAILED", nerrors
      else
        print *, "Test PASSED"
      endif

      close (unit=8)
      close (unit=9)

      end
