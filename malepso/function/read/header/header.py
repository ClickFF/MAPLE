def print_banner(output_file_name: str) -> None:
        """
        Print the MAPLE banner to the output file.
        """
        banner = r"""
**********************************************************************
*                                                                    *
*                        M   A   P   L   E                           *
*                                                                    *
*      MAchine-learning Potential for Landscape Exploration          *
*                                                                    *
*                                                                    *
*      © 2025 University of Pittsburgh. All rights reserved.         *
*      Licensed under CC BY 4.0 for academic use.                    *
*                                                                    *
*      Principal Developer:  Xujian Wang                             *
*                                                                    *
**********************************************************************

"""
        with open(output_file_name, 'a') as file:
            file.write(banner + "\n")
