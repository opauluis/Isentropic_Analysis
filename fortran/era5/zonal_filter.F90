! Zonal wavenumber decomposition along a full latitude circle
! Olivier Pauluis
!
! band_pass(w, k1, k2) returns the part of w(1:n) carried by zonal wavenumbers k1 to k2,
!   sum_k [a_k cos(2 pi k (i-1)/n) + b_k sin(2 pi k (i-1)/n)],
!   a_k = 2/n sum_m w_m cos(2 pi k (m-1)/n),  b_k = 2/n sum_m w_m sin(2 pi k (m-1)/n),
! which is the convolution with 2/n sum_k cos(2 pi k (i-m)/n) of the original code, in O(n (k2-k1+1))
! operations instead of O(n^2). Valid for 1 <= k1 <= k2 < n/2.

module zonal_filter
  implicit none
  private
  public :: band_pass

  integer, parameter :: dp = kind(1.0d0)

contains

  function band_pass(w, k1, k2) result(wf)
    real, intent(in) :: w(:)
    integer, intent(in) :: k1, k2
    real :: wf(size(w))

    real(dp), allocatable, save :: ctab(:,:), stab(:,:)
    integer, save :: n_tab = 0, k1_tab = 0, k2_tab = 0
    real(dp) :: a, b, pi
    integer :: n, i, k

    n = size(w)
    if (n /= n_tab .or. k1 /= k1_tab .or. k2 /= k2_tab) then
      if (allocated(ctab)) deallocate(ctab, stab)
      allocate(ctab(n, k1:k2), stab(n, k1:k2))
      pi = 4.0_dp * atan(1.0_dp)
      do k = k1, k2
        do i = 1, n
          ctab(i, k) = cos(2.0_dp * pi * k * (i - 1) / n)
          stab(i, k) = sin(2.0_dp * pi * k * (i - 1) / n)
        end do
      end do
      n_tab = n
      k1_tab = k1
      k2_tab = k2
    end if

    wf = 0.0
    do k = k1, k2
      a = 2.0_dp / n * dot_product(real(w, dp), ctab(:, k))
      b = 2.0_dp / n * dot_product(real(w, dp), stab(:, k))
      wf = wf + real(a * ctab(:, k) + b * stab(:, k))
    end do
  end function band_pass

end module zonal_filter
