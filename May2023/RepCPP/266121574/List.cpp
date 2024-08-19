template <typename type>
class List{
public:
List(){
this->num_elements = 0;
this->initialized = 0;
}

List(type input){
this->array = (type*)malloc(sizeof(type));
*(array) = input;
this->initialized = 1;
this->num_elements = 1;
}

void append(type item){
if(this->initialized){
size_t size = getSize();
this->array = (type*)realloc(this->array, size + 1*sizeof(type));

if(this->array == nullptr){
cout<<endl<<"Failed to realloc, freeing the array pointer"<<endl;
free(this->array);
}

*(this->array + size) = item;
this->num_elements += 1;

}

else{
this->array = (type*)malloc(sizeof(type));
*(array) = item;
this->initialized = 1;
this->num_elements += 1;
}

}

size_t getSize(){
auto size = this->num_elements;
return size;
}

type* data(int offset = 0){
assert(offset < this->getSize() - 1);

type* pointer = this->array + offset;
return pointer;
}



int remove(type item, int (*compare_func)(type v1, type v2) = nullptr, int remove_all=0){
auto size = (int)this->getSize();
int i;
std::queue<int> occurences;
if(compare_func == nullptr){
#pragma omp simd
for (i = 0; i<size; i++){
if(*(this->array + i) == item)
occurences.push(i);
}
if (remove_all){
int j = 0;
while(!occurences.empty()){
i = occurences.front();
std::copy(this->array + i + 1 -j , this->array + size - j, this->array + i -j);
this->num_elements -=1 ;
occurences.pop();
this->array = (type*)realloc(this->array, this->num_elements*(sizeof(type)));
j+=1;
}

}
else{
i = occurences.front();
std::copy(this->array + i + 1, this->array + size, this->array + i);
this->array = (type*)realloc(this->array, (size-1)*sizeof(type));
this->num_elements -= 1;
}
return 0;
}

else{
#pragma omp simd
for(i = 0; i<size; i++){
if(compare_func(*(this->array + i), item))
occurences.push(i);
}
if (remove_all){
int j = 0;
while(!occurences.empty()){
i = occurences.front();
std::copy(this->array + i + 1 -j, this->array + size - j, this->array + i -j);
occurences.pop();
this->num_elements -=1;
this->array = (type*)realloc(this->array, this->num_elements*(sizeof(type)));
j+=1;
}
}
else{
i = occurences.front();
std::copy(this->array + i + 1, this->array + size, this->array + i);
this->array = (type*)realloc(this->array, (size-1)*sizeof(type));
this->num_elements -= 1;
}
return 0;
}

return 1;
}

private:
type* array;
int initialized;
int num_elements;
};
