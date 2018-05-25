directories="deployments src openshift tests"

# checks for the whole directories
for directory in $directories
done
    grep -r -n "TODO: " $directory
done
