#include <iostream>
#include <limits>

void GetInt(int *in)
{
	if (!std::cin)
	{
		std::cin.clear();
		std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
	}

	std::cin>>*in;
}
