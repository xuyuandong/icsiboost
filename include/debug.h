/* Copyright (C) (2007) (Benoit Favre) <favre@icsi.berkeley.edu>

This program is free software; you can redistribute it and/or 
modify it under the terms of the GNU Lesser General Public License 
as published by the Free Software Foundation; either 
version 2 of the License, or (at your option) any later 
version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA. */

#ifndef __DEBUG_H__
#define __DEBUG_H__

namespace icsiboost{

#define DEBUG_INTERACTIVE 1
#define DEBUG_NON_INTERACTIVE 0

void init_debugging(const char* program_name, int interactive);

} // end namespace

#endif
