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
    real(dp), intent(in) :: w(:)
    integer, intent(in) :: k1, k2
    real(dp) :: wf(size(w))

    ! cos/sin tables for wavenumbers 1 to k_tab, kept between calls
    real(dp), allocatable, save :: ctab(:,:), stab(:,:)
    integer, save :: n_tab = 0, k_tab = 0
    real(dp) :: a, b, pi
    integer :: n, i, k

    n = size(w)
    if (n /= n_tab .or. k2 > k_tab) then
      if (allocated(ctab)) deallocate(ctab, stab)
      k_tab = max(k2, k_tab)
      allocate(ctab(n, k_tab), stab(n, k_tab))
      pi = 4.0_dp * atan(1.0_dp)
      do k = 1, k_tab
        do i = 1, n
          ctab(i, k) = cos(2.0_dp * pi * k * (i - 1) / n)
          stab(i, k) = sin(2.0_dp * pi * k * (i - 1) / n)
        end do
      end do
      n_tab = n
    end if

    wf = 0.0_dp
    do k = k1, k2
      a = 2.0_dp / n * dot_product(w, ctab(:, k))
      b = 2.0_dp / n * dot_product(w, stab(:, k))
      wf = wf + a * ctab(:, k) + b * stab(:, k)
    end do
  end function band_pass

end module zonal_filter
