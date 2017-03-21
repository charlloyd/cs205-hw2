!
! Copyright (c) 2016, NVIDIA Corporation.  All rights reserved.
!
! Please refer to the NVIDIA end user license agreement (EULA) associated
! with this source code for terms and conditions that govern your use of
! this software. Any use, reproduction, disclosure, or distribution of
! this software and related documentation outside the terms of the EULA
! is strictly prohibited.
!

module fp32

    real(kind=4) :: RISKFREE = 0.02
    real(kind=4) :: VOLATILITY = 0.30

contains
 
    real(kind=4)  function CND( d )
        implicit none
        real(kind=4) :: d
        real(kind=4), parameter :: A1 = 0.31938153
        real(kind=4), parameter :: A2 = -0.356563782
        real(kind=4), parameter :: A3 = 1.781477937
        real(kind=4), parameter :: A4 = -1.821255978
        real(kind=4), parameter :: A5 = 1.330274429
        real(kind=4), parameter :: RSQRT2PI = 0.39894228040143267793994605993438
        real(kind=4) :: K, abs, exp
    
        K = 1.0 / ( 1.0 + 0.2316419 * abs(d))
    
        CND = RSQRT2PI * exp(-0.5 * d * d) *          &
                (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
    
        if( d .gt. 0 ) CND = 1.0 - CND
    
        return
    end function

    subroutine blackscholesbody( callResult, putResult, Sf,Xf,Tf,Rf,Vf)
        implicit none
        real(kind=4) :: Sf, Xf, Tf, Rf, Vf
        real(kind=4) :: callResult, putResult
        real(kind=4) :: S, X, T, R, V
        real(kind=4) :: sqrtT, d1, d2, CNDD1, CNDD2, log, exp, expRT
    
        S = Sf
        X = Xf
        T = Tf
        R = Rf
        V = Vf
    
        sqrtT = sqrt(T)
        d1 = log( S / X ) + (R + 0.5 * V * V * T) / (V * sqrtT)
        d2 = d1 - V * sqrtT
        CNDD1 = CND( d1 )
        CNDD2 = CND( d2 )
    
        expRT = exp( -R * T)
        callResult = ( S * CNDD1 - X * expRT * CNDD2)
        putResult = (X * expRT * ( 1.0 - CNDD2) - S * (1.0 - CNDD1))
      
        return
    end subroutine

    subroutine blackscholes(h_callresult,h_putresult,h_stockprice, &
                            h_optionstrike,h_optionyears, riskfree, &
                            volatility, optN, accelerate)
        use accel_lib
        implicit none
        real(kind=4) :: h_callresult(0:*), h_putresult(0:*), h_stockprice(0:*)
        real(kind=4) :: h_optionstrike(0:*), h_optionyears(0:*), riskfree
        real(kind=4) :: volatility
        integer :: optN, accelerate, i

        real(kind=4) :: Sf, Xf, Tf, Rf, Vf
        real(kind=4) :: callResult, putResult
        real(kind=4) :: S, X, T, R, V
        real(kind=4) :: sqrtT, d1, d2, CNDD1, CNDD2, log, exp, expRT

        !$acc kernels do if (accelerate > 0)
        !$omp parallel do if (accelerate > 0)
        do i = 0, optN-1
            call blackscholesbody( h_callresult(i), h_putresult(i), &
            h_stockprice(i), h_optionstrike(i), h_optionyears(i), &
            riskfree, volatility )
        enddo
        !$acc end kernels
        !$end parallel do

    end subroutine

    real(kind=4) function randfloat( low, high )
        implicit none
        real(kind=4) :: low, high, t
        call random_number(t)
        randfloat = ( 1.0 - t ) * low + t * high
        return
    end function

end module fp32

program tester
    use fp32
    use accel_lib 
    implicit none
    integer opt_n, opt_sz, iterations, i
    real(kind=4),allocatable,dimension(:) :: callresultcpu, &
                putresultcpu,callresultgpu, putresultgpu, &
                stockprice,optionstrike, optionyears
    real(kind=4) :: delta, ref, sum_delta, sum_ref, max_delta 
    real(kind=4) :: l1norm, gputime, cpu_time
    real(kind=8) :: t(10), ms, msAcc
    
#ifdef _OPENACC
    call acc_init( acc_device_nvidia )
#endif
    
    opt_n = 4000000
    opt_sz = opt_n
    iterations = 10
    
    write(6,*)'Initializing data...'
    
    allocate( callresultcpu(0:opt_sz-1 ) )
    allocate( putresultcpu(0:opt_sz-1 ))
    allocate( callresultgpu(0:opt_sz-1 ))
    allocate( putresultgpu(0:opt_sz-1 ))
    allocate( stockprice(0:opt_sz-1 ))
    allocate( optionstrike(0:opt_sz-1 ))
    allocate( optionyears(0:opt_sz-1 ))
    
    ! generate options set
    
    do i = 0, opt_n-1
        callresultcpu(i) = 0.0
        putresultcpu(i) = -1.0
        callresultgpu(i) = 0.0
        putresultgpu(i) = -1.0
        stockprice(i) = randfloat( 5.0, 30.0 )
        optionstrike(i) = randfloat( 1.0, 100.0 )
        optionyears(i) = randfloat( 0.25, 10.0 )
    enddo
    
    write(6,*)''
    write(6,*)'Running Unaccelerated Version iterations ', iterations
    
    call cpu_time( t(1) )
    do i = 0, iterations-1
        call blackscholes( callresultcpu, putresultcpu,stockprice, &
                           optionstrike, optionyears, RISKFREE, &
                           VOLATILITY, opt_n, 0 )
    enddo
    call cpu_time( t(2) )
    ms = ( t(2) - t(1) ) / real( iterations, kind=8)
    
    write(6,*)''
    write(6,*)'Running Accelerated Version iterations ', iterations
    
    call cpu_time( t(3) )
    do i = 0, iterations-1
        call blackscholes( callresultgpu, putresultgpu,stockprice, &
                           optionstrike, optionyears, RISKFREE, &
                           VOLATILITY, opt_n, 1 )
    enddo
    call cpu_time( t(4) )
    msAcc = ( t(4) - t(3) ) / real( iterations, kind=8)
    
    write(6,*)''
    write(6,*)'Unaccelerated '
    write(6,*)'BlackScholes() time msec ', ms * 1.d3
    write(6,*)'GB/s ', (5.d0 * dble(opt_n) * 4 * 1.d-9) / (ms) 
    write(6,*)'GOptions/s ', (2.d0 * dble(opt_n) * 1.d-9) / (ms) 
    
    write(6,*)''
    write(6,*)'Accelerated '
    write(6,*)'BlackScholes() time msec ', msAcc * 1.d3
    write(6,*)'GB/s ', (5.d0 * dble(opt_n) * 4 * 1.d-9) / (msAcc) 
    write(6,*)'GOptions/s ', (2.d0 * dble(opt_n) * 1.d-9) / (msAcc) 
    
    write(6,*)''
    write(6,*)'Comparing the results...'
    
    sum_delta = 0.0
    sum_ref = 0.0
    max_delta = 0.0
    
    do i = 0, opt_n-1
        ref = callresultcpu(i)
        delta = abs( callresultcpu(i) - callresultgpu(i) )
        if( delta .gt. max_delta ) max_delta = delta
        sum_delta = sum_delta + delta
        sum_ref = sum_ref + abs( ref )
    enddo
    L1norm = sum_delta / sum_ref
    write(6,*)'L1 norm: ', L1norm
    write(6,*)'Max absolute error: ', max_delta

    if (max_delta .gt. 2.0e-5) then
       print *, "Test FAILED"
    else
       print *, "Test PASSED"
    endif
    
    deallocate( optionstrike(0:opt_sz-1 ))
    deallocate( optionyears(0:opt_sz-1 ))
    deallocate( stockprice )
    deallocate( putresultgpu )
    deallocate( putresultcpu )
    deallocate( callresultgpu )
    deallocate( callresultcpu )

end program
