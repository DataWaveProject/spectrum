!    Copyright (C) <2023>  <Yanmichel A. Morfa>
!
!    This program is free software: you can redistribute it and/or modify
!    it under the terms of the GNU General Public License as published by
!    the Free Software Foundation, either version 3 of the License, or
!    (at your option) any later version.
!
!    This program is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU General Public License for more details.
!
!    You should have received a copy of the GNU General Public License
!    along with this program.  If not, see <https://www.gnu.org/licenses/>.
!

!===================================================================================================
subroutine model_error(message)

    implicit none

    character (len = *), intent (in) :: message

    call model_message('Execution aborted with status ', ' Traceback: ', 1.d0, 'I2')
    call model_message(message, '.', 0.d0, 'I2')
    stop

    return
end subroutine model_error
!===================================================================================================

!===================================================================================================
subroutine model_message(main_message, optn_message, values, vfmt)

    implicit none

    character (len = *), intent(in) :: main_message
    character (len = *), intent(in) :: optn_message
    character (len = *), intent(in) :: vfmt

    double precision, intent(in) :: values
    character (len = 500) :: str_fmt
    integer :: ios

    str_fmt = '("' // trim(adjustl(main_message)) // '",' // trim(vfmt) // ',"' // &
            trim(adjustl(optn_message)) // '")'

    if (vfmt(1:1) == 'I') then
        write(unit = *, fmt = trim(adjustl(str_fmt)), iostat = ios) int(values)
    else
        write(unit = *, fmt = trim(adjustl(str_fmt)), iostat = ios) values
    endif

    if (ios /= 0) stop 'write error in unit '

    return
end subroutine model_message
!===================================================================================================

!===================================================================================================
integer function truncation(nspc)

    ! This function computes the triangular truncation given the number of spectral coefficients
    implicit none
    integer, intent(in) :: nspc

    ! compute triangular truncation
    truncation = int(-1.5 + 0.5 * sqrt(9. - 8. * (1. - float(nspc)))) + 1

    return
end function truncation
!===================================================================================================

!===================================================================================================
subroutine getspecindx(index_mn, ntrunc)

    ! This subroutine returns the spectral indices corresponding
    ! to the spherical harmonic degree l and the order m.

    implicit none

    integer, intent(in) :: ntrunc
    integer, intent(out) :: index_mn(2, (ntrunc + 1) * (ntrunc + 2) / 2)

    integer :: m, n, nmstrt, nm

    ! create spectral indices
    nmstrt = 0
    do m = 1, ntrunc + 1
        do n = m, ntrunc + 1
            nm = nmstrt + n - m + 1
            index_mn(:, nm) = [m, n]
        enddo
        nmstrt = nmstrt + ntrunc - m + 2
    enddo

    return
end subroutine getspecindx
!===================================================================================================

!===================================================================================================
subroutine onedtotwod(spec_2d, spec_1d, nlat, nspc, nt)

    implicit none
    ! input-output parameters
    integer, intent(in) :: nlat, nspc, nt
    double complex, intent(in) :: spec_1d(nspc, nt)
    double complex, intent(out) :: spec_2d(nlat, nlat, nt)
    ! local variables
    integer :: nmstrt, ntrunc, truncation
    integer :: n, m, mn

    ! compute triangular truncation. If nlat < ntrunc, then the coefficients are truncated
    ntrunc = truncation(nspc)

    ! initialize coefficients. If ntrunc < nlat - 1 the coefficients are filled with zeros
    spec_2d = 0.0

    nmstrt = 0
    do m = 1, nlat
        do n = m, nlat
            mn = nmstrt + n - m + 1
            spec_2d(m, n, :) = spec_1d(mn, :)
        enddo
        nmstrt = nmstrt + ntrunc - m + 1
    enddo

    return
end subroutine onedtotwod
!===================================================================================================

!===================================================================================================
subroutine twodtooned(spec_1d, spec_2d, nlat, ntrunc, nt)

    implicit none
    ! input-output parameters
    integer, intent(in) :: nlat, ntrunc, nt
    double complex, intent(in) :: spec_2d(nlat, nlat, nt)
    double complex, intent(out) :: spec_1d(ntrunc * (ntrunc + 1) / 2, nt)
    ! local variables
    integer :: nmstrt
    integer :: n, m, nm

    ! check number of coefficients (ntrunc <= nlat - 1)
    if (ntrunc > nlat) then
        call model_error("Bad number of coefficients: truncation > nlat - 1")
    end if

    ! initialize coefficients. If ntrunc < nlat - 1 the coefficients are filled with zeros
    spec_1d = 0.0

    nmstrt = 0
    do m = 1, ntrunc
        do n = m, ntrunc
            nm = nmstrt + n - m + 1
            spec_1d(nm, :) = spec_2d(m, n, :)
        enddo
        nmstrt = nmstrt + ntrunc - m + 1
    enddo

    return
end subroutine twodtooned
!===================================================================================================

!===================================================================================================
subroutine accumulate_order(spectrum, cs_lm, ntrunc, nspc, ns)
    ! input-output parameters
    implicit none

    integer, intent(in) :: ntrunc, nspc, ns
    double complex, intent(in) :: cs_lm     (nspc, ns)
    double precision, intent(out) :: spectrum  (ntrunc, ns)
    ! lcal variables
    double complex :: scaled_cs (ntrunc, ntrunc, ns)
    integer :: ln

    ! Reshape the spectral coefficients to matrix form (2, ntrunc, ntrunc, ...)
    call onedtotwod(scaled_cs, cs_lm, ntrunc, nspc, ns)

    ! Scale non-symmetric coefficients (ms != 1) by two
    scaled_cs(2:ntrunc, :, :) = 2.0 * scaled_cs(2:ntrunc, :, :)

    ! Initialize array for the 1D energy/power spectrum shaped (truncation, ...)
    spectrum = 0.0

    ! Compute spectrum as a function of total wavenumber: SUM Cml(m <= l).
    do ln = 1, ntrunc
        spectrum(ln, :) = real(sum(scaled_cs(1:ln, ln, :), dim = 1))
    enddo

    return
end subroutine accumulate_order
!===================================================================================================

!===================================================================================================
subroutine cross_spectrum(spectrum, clm_1, clm_2, ntrunc, nspc, ns)

    implicit none

    ! input-output parameters
    integer, intent(in) :: ntrunc, nspc, ns

    double complex, intent(in) :: clm_1(nspc, ns)
    double complex, intent(in) :: clm_2(nspc, ns)
    double precision, intent(out) :: spectrum(ntrunc, ns)

    ! lcal variables
    double complex :: clm_cs   (nspc, ns)

    ! Compute cross spectrum in (m, l) space
    clm_cs = clm_1 * conjg(clm_2)

    ! Compute spectrum as a function of total wavenumber: SUM Cml(m <= l).
    call accumulate_order(spectrum, clm_cs, ntrunc, nspc, ns)

    return
end subroutine cross_spectrum
!===================================================================================================

